
#--------------------------------------------- Experiment 1 ---------------------------------------------------------------

for param in self.net_g.to_feat.parameters():
    param.requires_grad = False

print("Freezing the Attention Blocks ..........")
for i in range(4):  # Adjust this number based on how many blocks you want to freeze
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = False
for i in range(4, len(self.net_g.feats)):  # Unfreeze the last 2 blocks
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = True
for param in self.net_g.to_img.parameters():
    param.requires_grad = True


#--------------------------------------------- Experiment 2 ---------------------------------------------------------------

for param in self.net_g.to_feat.parameters():
    param.requires_grad = False

print("Freezing the Attention Blocks ..........")
for i in range(2):  # Adjust this number based on how many blocks you want to freeze
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = True
for i in range(2, len(self.net_g.feats)):  # Unfreeze the last 2 blocks
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = False
for param in self.net_g.to_img.parameters():
    param.requires_grad = True


#--------------------------------------------- Experiment 3 ---------------------------------------------------------------

for param in self.net_g.to_feat.parameters():
    param.requires_grad = False

print("Freezing  the Attention Blocks ..........")
for i in range(7):  # Adjust this number based on how many blocks you want to freeze
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = False
for i in range(7, len(self.net_g.feats)):  # Unfreeze the last 2 blocks
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = True
for param in self.net_g.to_img.parameters():
    param.requires_grad = True


# Modified lmlt_arch.py 
drop_rate = 0.1
attn_drop_rate = 0.1


#--------------------------------------------- Experiment 4 ---------------------------------------------------------------


for param in self.net_g.to_feat.parameters():
    param.requires_grad = False

print("Freezing  the Attention Blocks ..........")
for i in range(6):  # Adjust this number based on how many blocks you want to freeze
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = False
for i in range(6, len(self.net_g.feats)):  # Unfreeze the last 2 blocks
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = True
for param in self.net_g.to_img.parameters():
    param.requires_grad = True



# Modified Learning rate : 1e-5



#--------------------------------------------- Experiment 5 ---------------------------------------------------------------


for param in self.net_g.to_feat.parameters():
    param.requires_grad = False

print("Freezing  the Attention Blocks ..........")
for i in range(8):  # Adjust this number based on how many blocks you want to freeze
    for param in self.net_g.feats[i].parameters():
        param.requires_grad = False
for param in self.net_g.to_img.parameters():
    param.requires_grad = True



# Modified Learning rate : 5e-6


#--------------------------------------------- Experiment 6 ,7 ,8---------------------------------------------------------------

# Knowledge Distillation with distillation loss with variations 












