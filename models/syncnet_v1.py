import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d
from .convnext import ConvNeXt

class SyncNet_ConvNext(nn.Module):
    def __init__(self):
        super(SyncNet_ConvNext, self).__init__()

        self.face_encoder = nn.Sequential(
            ConvNeXt(15, dims=(32, 64, 128, 512), kernel_sizes=5, patch_size=3),
            Conv2d(512, 512, kernel_size=3, stride=(1,2), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
        )

        self.audio_encoder = nn.Sequential(
            ConvNeXt(1, dims=(16, 32, 64, 256), kernel_sizes=5, patch_size=1),
            Conv2d(256, 512, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(2,2), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(2,1), padding=1),
            )
            
    def forward(self, audio_sequences, face_sequences):# audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        face_embedding = torch.reshape(face_embedding, (face_embedding.size(0), -1))
        audio_embedding = torch.reshape(audio_embedding, (audio_embedding.size(0), -1))
        
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)


        return audio_embedding, face_embedding


''' Testing Syncnet_ConvNext file for face_embedding'''
# sync_nxt = SyncNet_ConvNext()
# mel = torch.rand(1, 15, 80, 160)
# print(mel.shape)
# print(sync_nxt(mel).shape) #only for face_embedding [1, 4608]