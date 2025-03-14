import itertools
import numpy as np
import torch
from tqdm import tqdm
from baseline.binet.core import NNAnomalyDetector
from baseline.comb.components.decoder import Decoder
from baseline.comb.components.encoder import Encoder
from novel.utils.component_runtime_tracker import ComponentRuntimeTracker
from utils.dataset import Dataset
from torch.utils.data import TensorDataset, DataLoader

from utils.eval import cal_best_PRF

# https://github.com/guanwei49/COMB
class COMB(NNAnomalyDetector):
    """Implements a transformer based anomaly detection algorithm."""

    abbreviation = 'comb'
    name = 'COMB'
    supports_attributes = True

    # COMB taken as baseline hyperparameter configuration
    config = dict(         
        # model parameter setting
        batch_size = 64,
        d_model = 64,
        n_layers_agg = 2,
        n_layers = 2,
        n_heads = 4,
        ffn_hidden = 128,
        drop_prob = 0.1,

        n_epochs = 20,
        lr = 0.0002,
        b1 = 0.5,
        b2 = 0.999,
    ) 

    def __init__(self, config):
        super(COMB, self).__init__(config)

        # Update the config with the additional parameters
        self.config.update(config)

    def build_model(self, attribute_dims, max_len, device):
        d_model = self.config.get('d_model',64)
        n_layers_agg = self.config.get('n_layers_agg',2)
        n_layers = self.config.get('n_layers',2)
        n_heads = self.config.get('n_heads',4)
        ffn_hidden = self.config.get('ffn_hidden',128)
        drop_prob = self.config.get('drop_prob',0.1)

        encoder = Encoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob, device)
        decoder = Decoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)

        return encoder, decoder
    
    def train(self, encoder:Encoder, decoder:Decoder, dataloader, device, attribute_dims):
        n_epochs = self.config.get('n_epochs',20)
        lr = self.config.get('lr',0.0002)
        b1 = self.config.get('b1',0.5)
        b2 = self.config.get('b2',0.999)

        optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(),decoder.parameters()),lr=lr, betas=(b1, b2))
        step_size = max(1, int(n_epochs/2))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

        print("*"*10+"training"+"*"*10)
        for epoch in range(int(n_epochs)):
            train_loss = 0.0
            train_num = 0
            for i, Xs in enumerate(tqdm(dataloader)):
                mask = Xs[-1]
                Xs = Xs[:-1]
                mask = mask.to(device)
                for k ,X in enumerate(Xs):
                    Xs[k] = X.to(device)

                optimizer.zero_grad()

                enc_output = encoder(Xs, mask)
                fake_X = decoder(Xs, enc_output,mask)

                loss = 0.0
                for ij in range(len(attribute_dims)):
                    # --------------
                    # 除了每一个属性的起始字符之外,其他重建误差
                    # ---------------
                    pred = torch.softmax(fake_X[ij][:, :-1, :], dim=2).flatten(0, -2) #最后一个预测无意义
                    true = Xs[ij][:, 1:].flatten()

                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                            fake_X[0].shape[1] - 1)

                    cross_entropys = -torch.log(corr_pred)
                    loss += cross_entropys.masked_select((mask[:, 1:])).mean()

                train_loss += loss.item() * Xs[0].shape[0]
                train_num +=Xs[0].shape[0]
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch=train_loss / train_num
            print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
                f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return encoder,decoder        

    def detect(self, encoder:Encoder, decoder:Decoder, dataloader, device, attribute_dims, attr_Shape):
        encoder.eval()
        decoder.eval()
        pos=0
        with torch.no_grad():
            attr_level_abnormal_scores=np.zeros(attr_Shape)

            print("*" * 10 + "detecting" + "*" * 10)

            for Xs in tqdm(dataloader):
                mask = Xs[-1]
                Xs = Xs[:-1]
                for k,tempX in enumerate(Xs):
                    Xs[k] = tempX.to(device)
                mask=mask.to(device)

                enc_output = encoder(Xs, mask)
                fake_X = decoder(Xs, enc_output, mask)

                for attr_index in range(len(attribute_dims)):
                    fake_X[attr_index]=torch.softmax(fake_X[attr_index][:, :-1, :],dim=2)

                #求异常分数 (Calculate anomaly scores)
                for attr_index in range(len(attribute_dims)):
                    truepos = Xs[attr_index][:, 1:].flatten()
                    p = fake_X[attr_index].reshape((truepos.shape[0],-1)).gather(1, truepos.view(-1, 1)).squeeze()
                    p_distribution = fake_X[attr_index].reshape((truepos.shape[0],-1))

                    p_distribution = p_distribution + 1e-8  # 避免出现概率为0 (Avoid probabilities of 0)

                    attr_level_abnormal_scores[pos: pos + Xs[attr_index].shape[0], 1: ,attr_index] = \
                        ((torch.sum(torch.log(p_distribution) * p_distribution, 1) - torch.log(p)).reshape((Xs[attr_index].shape[0],-1))*(mask[:,1:])).detach().cpu()
                pos += Xs[attr_index].shape[0]

            trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
            return trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
        
    def temp_post_process(self, trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores, trace_labels, event_labels, attr_labels):
        ##trace level
        trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(trace_labels, trace_level_abnormal_scores)
        print("trace-level", trace_f1)
        # print("Precision:{}, Recall:{}, F-score:{}, AP:{}".format(trace_p, trace_r, trace_f1, trace_aupr))

        ##event level
        event_p, event_r, event_f1, event_aupr = cal_best_PRF(event_labels.flatten(), event_level_abnormal_scores.flatten())
        print("event-level", event_f1)
        # print("Precision:{}, Recall:{}, F-score:{}, AP:{}".format(event_p, event_r, event_f1, event_aupr))

        ##attr level
        attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(attr_labels.flatten(), attr_level_abnormal_scores.flatten())
        print("attr-level", attr_f1)
        # print("Precision:{}, Recall:{}, F-score:{}, AP:{}".format(attr_p, attr_r, attr_f1, attr_aupr))

        return trace_f1, event_f1, attr_f1        

    def train_and_predict(self, dataset:Dataset):
        component_runtime_tracker:ComponentRuntimeTracker = self.config.get('component_runtime_tracker', None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare the dataset
        component_runtime_tracker.start_component('prepare_data')
        
        attribute_dims = dataset.attribute_dims
        max_len = dataset.max_len

        Xs=[]
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append( torch.LongTensor(dataset.features[i]))
        mask = torch.BoolTensor(dataset.mask)
        train_Dataset = TensorDataset(*Xs, mask)
        detect_Dataset = TensorDataset(*Xs, mask)

        batch_size = self.config.get('batch_size', 64)

        train_dataloader = DataLoader(train_Dataset, batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)
        detect_dataloader = DataLoader(detect_Dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

        component_runtime_tracker.end_component('prepare_data')

        # Prepare the COMB model
        component_runtime_tracker.start_component('build_model')

        encoder, decoder = self.build_model(attribute_dims, max_len, device)
        encoder.to(device)
        decoder.to(device)

        component_runtime_tracker.end_component('build_model')

        # Train the model
        component_runtime_tracker.start_component('train_predict_model')
        encoder, decoder = self.train(encoder, decoder, train_dataloader, device, attribute_dims)

        # Detect anomalies
        attr_Shape=(dataset.num_cases, dataset.max_len, dataset.num_attributes)
        trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = self.detect(encoder, decoder, detect_dataloader, device, attribute_dims, attr_Shape)
        component_runtime_tracker.end_component('train_predict_model')
        
        # Post-process the results
        component_runtime_tracker.start_component('post_process_results')
        
        trace_level_abnormal_scores = np.asarray(trace_level_abnormal_scores)
        event_level_abnormal_scores = np.asarray(event_level_abnormal_scores)
        attr_level_abnormal_scores = np.asarray(attr_level_abnormal_scores)

        single_OA_case_labels = dataset.case_labels[:, :2]
        single_OA_event_labels = dataset.event_labels[:, :, :2]
        single_OA_attr_labels = dataset.attr_labels[:, :, :, :2]

        single_output_case_labels = np.max(dataset.case_labels, axis=-1)
        single_output_event_labels = np.max(dataset.event_labels, axis=-1)
        single_output_attr_labels = np.max(dataset.attr_labels, axis=-1)

        single_OA_output_case_labels = np.max(single_OA_case_labels, axis=-1)
        single_OA_output_event_labels = np.max(single_OA_event_labels, axis=-1)
        single_OA_output_attr_labels = np.max(single_OA_attr_labels, axis=-1)

        component_runtime_tracker.end_component('post_process_results')

        print("single_output")
        trace_f1, event_f1, attr_f1 = self.temp_post_process(
            trace_level_abnormal_scores, 
            event_level_abnormal_scores, 
            attr_level_abnormal_scores, 
            single_output_case_labels, 
            single_output_event_labels, 
            single_output_attr_labels)
        print("single_OA_output")
        trace_OA_f1, event_OA_f1, attr_OA_f1 = self.temp_post_process(
            trace_level_abnormal_scores, 
            event_level_abnormal_scores, 
            attr_level_abnormal_scores, 
            single_OA_output_case_labels, 
            single_OA_output_event_labels, 
            single_OA_output_attr_labels)

        # print("trace_level_abnormal_scores")
        # print(dataset.case_labels.shape)
        # print(trace_level_abnormal_scores.shape)
        # print("event_level_abnormal_scores")
        # print(dataset.event_labels.shape)
        # print(event_level_abnormal_scores.shape)
        # print("attr_level_abnormal_scores")
        # print(dataset.attr_labels.shape)
        # print(attr_level_abnormal_scores.shape)

        self.config['COMB_results'] = {
            'trace_f1': trace_f1,
            'event_f1': event_f1,
            'attr_f1': attr_f1,
            'trace_OA_f1': trace_OA_f1,
            'event_OA_f1': event_OA_f1,
            'attr_OA_f1': attr_OA_f1
        }

        attribute_perspectives = dataset.event_log.event_attribute_perspectives
        attribute_names = dataset.event_log.event_attribute_keys
        results = ({0:trace_level_abnormal_scores}, 
                {0:event_level_abnormal_scores}, 
                {0:attr_level_abnormal_scores},
                {0:None},  
                {0:single_OA_output_case_labels},
                {0:single_OA_output_event_labels}, 
                {0:single_OA_output_attr_labels}, 
                {0:None},
                attribute_perspectives,
                attribute_perspectives,
                attribute_names,
                attribute_names,
                None,
                None,
                None)

        return results

        




        
        


    