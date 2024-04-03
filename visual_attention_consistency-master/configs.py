
# Dataset configurations
datasets = ["MINE"]

def get_configs(dataset):
	opts = {}
	if not dataset in datasets:
		raise Exception("Not supported dataset!")
	else:
		if dataset == "MINE":
			opts["dataset_path"] = r"E:\MY_DATASET\CLASSIFICATION"
			opts["num_labels"] = 4
			opts["scale"] = [256, 512]
		else:
			pass
	return opts