from CRPN import CRPNProbing

crpn = CRPNProbing('/mnt/sda/luhan/ViT-pytorch-main/data/FED-RO_crop')
df = crpn.evaluate('predict')
acc = (df['pred'] == df['label']).mean()
print(acc)

