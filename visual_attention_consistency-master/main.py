import torch
import torch.optim as optim

from test import test, validate
from configs import get_configs
from utils import *
import argparse
import time
import os
import ast


def get_parser():
	parser = argparse.ArgumentParser(description = 'CNN Attention Consistency')
	parser.add_argument("--dataset", default="MINE", type=str,
						help="select a dataset to train models")
	parser.add_argument("--dataset_path", default=r"E:\MY_DATASET\CLASSIFICATION(ENHANCE)", type=str,
		help="select the dataset path to train models")
	parser.add_argument("--img_scale", default=(256, 512),
						type=lambda x: tuple(map(int, ast.literal_eval(x))),
						help="Input the scale of the img into dataset")
	parser.add_argument("--arch", default='hybrid4', type=str,
		help="ResNet architecture")

	parser.add_argument('--train_batch_size', default=4, type=int,
		help='default training batch size')
	parser.add_argument('--train_workers', default=4, type=int,
		help='# of workers used to load training samples')
	parser.add_argument('--val_batch_size', default=4, type=int,
						help='default val batch size')
	parser.add_argument('--val_workers', default=4, type=int,
						help='# of workers used to load validating samples')
	parser.add_argument('--test_batch_size', default=4, type=int,
		help='default test batch size')
	parser.add_argument('--test_workers', default=4, type=int,
		help='# of workers used to load testing samples')

	parser.add_argument('--learning_rate', default=3e-2, type=float,
		help='base learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,
		help="set the momentum")
	parser.add_argument('--weight_decay', default=0.0005, type=float,
		help='set the weight_decay')
	parser.add_argument('--stepsize', default=1, type=int,
		help='lr decay each # of epoches')
	parser.add_argument('--decay', default=0.5, type=float,
		help='update learning rate by a factor')

	parser.add_argument('--model_dir', default='checkpoint/models',
		type=str, help='path to save checkpoints')
	parser.add_argument('--model_prefix', default='model',
		type =str, help='model file name starts with')

	# optimizer
	parser.add_argument('--optimizer', default='SGD',
		type=str, help='Select an optimizer: TBD')

	# general parameters
	parser.add_argument('--epoch_max', default=1, type=int,
		help='max # of epcoh')
	parser.add_argument('--display', default=20, type=int,
		help='display')
	parser.add_argument('--snapshot', default=1, type=int,
		help='snapshot')
	parser.add_argument('--start_epoch', default=0, type=int,
		help='resume training from specified epoch')
	parser.add_argument('--pretrained', default=False, type=bool,
		help='load preatrained params before training')
	parser.add_argument('--resume', default=r'F:\visual_attention_consistency-master\visual_attention_consistency-master\checkpoint\models\restv2_mine\2-27\val_epoch16.pth', type=str,
		help='resume training from specified model state')

	parser.add_argument('--val', default=True, type=bool,
		help='conduct validating after each checkpoint being saved')
	parser.add_argument('--test', default=True, type=bool,
		help='conduct testing the best validation checkpoint')


	return parser


def main():
	parser = get_parser()
	print(parser)
	args = parser.parse_args()
	print(args)

	# load data
	opts = get_configs(args.dataset)
	print(opts["dataset_path"])

	print(opts)
	trainset, valset, testset = get_dataset(opts)

	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size=args.train_batch_size,
		shuffle=False,
		num_workers=args.train_workers)

	val_loader = torch.utils.data.DataLoader(valset,
	    batch_size=args.val_batch_size,
	    shuffle=False,
	    num_workers=args.val_workers)

	test_loader=torch.utils.data.DataLoader(testset,
		batch_size=args.test_batch_size,
		shuffle=False,
		num_workers=args.test_workers)


	# path to save models
	if not os.path.isdir(args.model_dir):
		print("Make directory: " + args.model_dir)
		os.makedirs(args.model_dir)

	# prefix of saved checkpoint
	model_prefix = args.model_dir + '/' + args.model_prefix

	# ---------------------- ResNet ----------------------
	if args.arch == "resnet18":
		from resnet import resnet18
		model = resnet18(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "resnet34":
		from resnet import resnet34
		model = resnet34(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "resnet50":
		from resnet import resnet50
		model = resnet50(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "resnet101":
		from resnet import resnet101
		model = resnet101(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "resnet152":
		from resnet import resnet152
		model = resnet152(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_" + args.arch

	# ---------------------- VGG16 ----------------------
	elif args.arch == "vgg16":
		from model.VGG16 import VGG
		model = VGG()
		model_prefix = model_prefix + "_" + args.arch

	# ---------------------- MobileNet ----------------------
	elif args.arch == "mobilenet":
		from torchvision.models import MobileNetV2
		model = MobileNetV2(num_classes=4)
		model_prefix = model_prefix + "_" + args.arch

	# --------------------- EfficientNet ---------------------
	elif args.arch == "efficientnet0":
		from torchvision.models import efficientnet_b0
		model = efficientnet_b0(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet1":
		from torchvision.models import efficientnet_b1
		model = efficientnet_b1(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet2":
		from torchvision.models import efficientnet_b2
		model = efficientnet_b2(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet3":
		from torchvision.models import efficientnet_b3
		model = efficientnet_b3(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet4":
		from torchvision.models import efficientnet_b4
		model = efficientnet_b4(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet5":
		from torchvision.models import efficientnet_b5
		model = efficientnet_b5(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet65":
		from torchvision.models import efficientnet_b6
		model = efficientnet_b6(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "efficientnet7":
		from torchvision.models import efficientnet_b7
		model = efficientnet_b7(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch

	# ----------------------- ConvNeXt -----------------------
	elif args.arch == "convnext_tiny":
		from torchvision.models import convnext_tiny
		model = convnext_tiny(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "convnext_small":
		from torchvision.models import convnext_small
		model = convnext_small(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "convnext_base":
		from torchvision.models import convnext_base
		model = convnext_base(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "convnext_large":
		from torchvision.models import convnext_large
		model = convnext_large(weights=None, num_classes=4)
		model_prefix = model_prefix + "_" + args.arch

	# ----------------------- DenseNet -----------------------
	elif args.arch == "densenet":
		from model.DenseNet import DenseNet
		model = DenseNet()
		model_prefix = model_prefix + "_" + args.arch

	# ---------------------- MINE ----------------------
	elif args.arch == "django":
		from model.MINE.django import Django_Class_Net
		model = Django_Class_Net()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv2_mine":
		from model.restv2 import restv2_mine
		model = restv2_mine()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "convformer":
		from model.MINE.convformer import Setr_ConvFormer
		model = Setr_ConvFormer()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "res_conv_t":
		from model.MINE.res_conv_t import res_conv_t
		model = res_conv_t()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid_new":
		from model.MINE.Hybrid_new import Hybrid_new
		model = Hybrid_new()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid2":
		from model.MINE.hybrid2 import resnet_hybrid
		model = resnet_hybrid(pretrained=False, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid3":
		from model.MINE.hybrid3 import hybrid3
		model = hybrid3()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid4":
		from model.MINE.hybrid4 import hybrid4
		model = hybrid4()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid4":
		from model.MINE.hybrid4 import hybrid4
		model = hybrid4()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid44":
		from model.MINE.hybrid44 import hybrid44
		model = hybrid44()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "hybrid5":
		from model.MINE.hybrid5 import hybrid5
		model = hybrid5()
		model_prefix = model_prefix + "_" + args.arch

	# ------------------ Res Transformer ------------------
	elif args.arch == "restv1_lite":
		from model.restv1 import rest_lite
		model = rest_lite()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv1_small":
		from model.restv1 import rest_small
		model = rest_small()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv1_base":
		from model.restv1 import rest_base
		model = rest_base()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv1_large":
		from model.restv1 import rest_large
		model = rest_large()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv2_tiny":
		from model.restv2 import restv2_tiny
		model = restv2_tiny()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv2_small":
		from model.restv2 import restv2_small
		model = restv2_small()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv2_base":
		from model.restv2 import restv2_base
		model = restv2_base()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "restv2_large":
		from model.restv2 import restv2_large
		model = restv2_large()
		model_prefix = model_prefix + "_" + args.arch


	# ----------------- Shunted Transformer -----------------
	elif args.arch == "shuntedt_t":
		from model.ShuntedT import shunted_t
		model = shunted_t()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "shuntedt_s":
		from model.ShuntedT import shunted_s
		model = shunted_s()
		model_prefix = model_prefix + "_" + args.arch
	elif args.arch == "shuntedt_b":
		from model.ShuntedT import shunted_b
		model = shunted_b()
		model_prefix = model_prefix + "_" + args.arch

	# ------------------ Swin Transformer ------------------
	elif args.arch == "swint":
		from model.SwinT import SwinTransformer
		model = SwinTransformer()
		model_prefix = model_prefix + "_" + args.arch

	# ----------------- Vision Transformer -----------------
	elif args.arch == "vit":
		from model.VIT import ViT
		model = ViT()
		model_prefix = model_prefix + "_" + args.arch

	# ----------------- FITNet -----------------
	elif args.arch == "fitnet":
		from model.fitnet import FITNet
		model = FITNet()
		model_prefix = model_prefix + "_" + args.arch

	# ----------------- Med ViT -----------------
	elif args.arch == "medvit":
		from model.medvit import MedViT_small
		model = MedViT_small()
		model_prefix = model_prefix + "_" + args.arch

	else:
		raise NotImplementedError("To be implemented!")

	if args.pretrained:
		resume_model = torch.load(args.resume)
		model.load_state_dict(resume_model)

	# print(model)
	model.cuda()

	if args.optimizer == 'Adam':
		optimizer = optim.Adam(
			model.parameters(),
			lr=args.learning_rate
		)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(
			model.parameters(),
			lr=args.learning_rate,
			momentum=args.momentum,
			weight_decay=args.weight_decay
		)
	else:
		raise NotImplementedError("Not supported yet!")

	#pretrained_weights_path = r".\checkpoint\models\restv2_mine\2-27\val_epoch16.pth"
	#pretrained_weights = torch.load(pretrained_weights_path)
	#model.load_state_dict(pretrained_weights)
	# training the network
	model.train()

	criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
	criterion3 = Recall_loss

	best_recall = 0.0
	best_epoch = 0
	model_dir = f"checkpoint/models/{args.arch}"
	for epoch in range(args.start_epoch, args.epoch_max):
		epoch_start = time.time()
		if not args.stepsize == 0:
			adjust_learning_rate(optimizer, epoch, args)
		for batch, (img, label) in enumerate(train_loader):
			img = img.cuda()
			label = label.cuda()
			output = model(img)
			output = torch.sigmoid(output)
			loss = criterion(output, label)
			#loss = criterion(output, label) + 0.2 * criterion3(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (batch) % args.display == 0:
				print(
					'epoch: {},\ttrain step: {}\tLoss: {:.6f}'.format(epoch+1,
					batch, loss.item())
				)

		epoch_end = time.time()
		elapsed = epoch_end - epoch_start
		print("Epoch time: ", elapsed)

		# validate
		if (epoch+1) % args.snapshot == 0:

			if args.val:
				model.eval()
				val_start = time.time()
				epoch_recall, _epoch = validate(model, val_loader, epoch+1)
				val_time = (time.time() - val_start)
				print("val time: ", val_time)
				if epoch_recall > best_recall:
					best_recall = epoch_recall
					best_epoch = _epoch

					if not os.path.exists(model_dir):
						os.makedirs(model_dir)
					torch.save(model.state_dict(), os.path.join(model_dir, f"val_epoch{best_epoch}.pth"))

	current_directory = os.getcwd()
	best_model_path = f"checkpoint/models/{args.arch}/val_epoch{best_epoch}.pth"
	full_path = os.path.join(current_directory, best_model_path)
	full_path = r"./duibi2/mine/5(wo ga)/val_best(√).pth"
	print(full_path)
	if epoch + 1 == args.epoch_max:
		if os.path.exists(full_path):
			print(f"Now the test is been loaded at epoch{best_epoch} checkpoint")
			model.load_state_dict(torch.load(full_path))
			model.eval()
			test(model, test_loader, args.arch)
			torch.save(model.state_dict(), f"checkpoint/models/{args.arch}/val_best.pth")




if __name__ == '__main__':
	main()
