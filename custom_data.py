import torch 
import torchaudio

class AudioDataset:
    def __init__(self, file_path, class_id):
        self.file_path = file_path
        self.class_id = class_id
        
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        path = self.file_path[idx]
        waveform, sr = torchaudio.load(path) # load audio
        audio_mono = torch.mean(waveform, dim=0, keepdim=True) # Convert sterio to mono
        tempData = torch.zeros([1, sr*4])
        if audio_mono.numel() < sr*4: # if sample_rate < 160000
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :sr*4] # else sample_rate 160000
        audio_mono=tempData

        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono) # (channel, n_mels, time)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std() # Noramalization
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono) # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std() # mfcc norm
        
        new_feat = torch.cat([mel_specgram_norm, mfcc_norm], axis=1)

        return {
            "specgram": new_feat[0].permute(1, 0),
            "label": torch.tensor(self.class_id[idx], dtype=torch.long)
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(data):
    specs = []
    labels = []
    for d in data:
        spec = d["specgram"]
        label = d["label"]
        specs.append(spec)
        labels.append(label)
    spec = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True, padding_value=0.)
    labels = torch.tensor(labels)
    return spec, labels