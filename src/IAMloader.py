import os
import glob
import cv2
import tensorflow as tf
import random


# Class for loading filtered datasets from the 'words' set in the IAM Handwriting
# database. The argument 'datapath' expects to receive a path to the directory
# containing a two things: a subfolder 'words' that has all the images for the 
# dataset as PNG files in no particular organization (globs all files recursively
# regardless of subdir or whatever), and a subfolder 'ascii' that contains the
# 'words.txt' file provided by IAM.
class WordsLoader:
	def __init__(self, datapath=os.path.normpath('../data/iam/'), verbose=False):
		self.datapath = datapath
		self.words_dir_path = os.path.join(self.datapath,'words')
		self.words_txt_path = os.path.join(self.datapath,'ascii/words.txt')

		self.attributes = ['id','clean','binarizationThreshold','x','y','w','h','tag','transcription','path']

		self.v = verbose

		assert os.path.isdir(self.datapath), 'need /data at toplevel of repo'
		assert os.path.isdir(self.words_dir_path), 'need /words directory in /data directory'
		assert os.path.isfile(self.words_txt_path), 'need /ascii/words.txt file in /data directory'

		self.data = {}
		self._unfiltered_data = {}

	def __len__(self):
		return len(self.data)

	# processes words.txt file and associates that index with the files in the words/ dir
	def index(self):
		self.data = {}
		self._unfiltered_data = {}

		with open(self.words_txt_path, mode='r') as f:
			if self.v:
				print(f'Reading file {self.words_txt_path} ...',end='')
			raw = f.read()
			if self.v:
				print(' done!')

		if self.v:
			print('Processing words.txt info...',end='')

		for line in raw.split('\n'):
			if line and (not line[0] == '#'):
				l = line.split(' ')
				if len(l) != 9:
					continue

				self.data[l[0]] = {
					self.attributes[0]:  l[0],
					self.attributes[1]: True if (l[1] == 'ok') else False,
					self.attributes[2]: int(l[2]),
					self.attributes[3]: int(l[3]),
					self.attributes[4]: int(l[4]),
					self.attributes[5]: int(l[5]),
					self.attributes[6]: int(l[6]),
					self.attributes[7]: l[7],
					self.attributes[8]: l[8],
					self.attributes[9]: None
				}

		if self.v:
				print(' done!')

		if self.v:
			print(f"Finding all PNG files in {self.words_dir_path} ...", end='')

		image_paths = glob.glob(self.words_dir_path + '/**/*.png', recursive=True)

		if self.v:
			print(f" done!")

		if self.v:
			print(f"Associating all found image files with indices...")

		for p in image_paths:
			try:
				basename_no_ext = os.path.basename(p)[:-4]
				datum = self.data[basename_no_ext]
				datum['path'] = os.path.abspath(p)
			except KeyError as e:
				if self.v:
					print('No matching datum in loader.data for file name: '+str(p))
				continue

		removed_count = 0
		keys = list(self.data.keys())
		for k in keys:
			if not self.data[k]['path']:
				del self.data[k]
				removed_count += 1

		self._unfiltered_data = self.data.copy()

		print(f'Indexing done! {len(self.data)} entries indexed.')
		print(f'Could not find associated images for {removed_count} entries in words.txt.')

	# returns data to unfiltered state
	def clear_filters(self):
		self.data = self._unfiltered_data.copy()

	# returns count of data whose attribute meet condition
	def count(self, attribute, condition):
		assert attribute in self.attributes, "attribute must be in "+str(self.attributes)+"."
		
		count = 0
		for k in self.data.keys():
			if condition(self.data[k][attribute]):
				count += 1
		
		return count

	# returns list of ids of data whose attribute meet condition
	def ids_where(self, attribute, condition):
		assert attribute in self.attributes, "attribute must be in "+str(self.attributes)+"."
		
		ids = []
		for k in self.data.keys():
			if condition(self.data[k][attribute]):
				ids.append(self.data[k]['id'])
		
		return ids

	# internally filters out data whose attribute does not meet condition
	def filter(self, attribute, condition):
		assert attribute in self.attributes, "attribute must be in "+str(self.attributes)+"."
		
		new_data = {}
		for k in self.data.keys():
			if condition(self.data[k][attribute]):
				new_data[k] = self.data[k].copy()
		self.data = new_data

	# splits data into training, validation, and testing sets, loads image data,
	# and converts everything to three separate TF Datasets. Can perform batching,
	# binarization, and shuffling if specified. pass '0' to split args if not using 
	# validation or testing set. 
	def load(self, valid_split=0.2, test_split=0.1, shuffle=True, binarize=True, batch_size=0):
		train_split = 1 - valid_split - test_split
		assert 1 >= train_split > 0 
		assert 1 >= valid_split >= 0
		assert 1 >= test_split >= 0
		assert round(train_split + valid_split + test_split) == 1

		count = len(self.data)

		if valid_split and test_split:
			split_idx_1 = int(count * train_split)
			split_idx_2 = int(count * valid_split) + split_idx_1
		elif valid_split:
			split_idx_1 = int(count * train_split)
			split_idx_2 = count
		elif test_split:
			split_idx_1 = int(count * train_split)
			split_idx_2 = split_idx_1
		else:
			split_idx_1 = split_idx_2 = count

		index = list(self.data.values())

		binarization_thresholds = []
		transcriptions = []
		image_paths = []

		max_w = index[0]['w']
		max_h = index[0]['h']
		for item in index:
			if item['w'] > max_w:
				max_w = item['w']
			if item['h'] > max_h:
				max_h = item['h']

			binarization_thresholds.append(item['binarizationThreshold'])
			transcriptions.append(item['transcription'])
			image_paths.append(item['path'])

		if shuffle:
			temp = list(zip(binarization_thresholds, transcriptions, image_paths)) 
			random.shuffle(temp) 
			binarization_thresholds, transcriptions, image_paths = zip(*temp)
			binarization_thresholds = list(binarization_thresholds)
			transcriptions = list(transcriptions)
			image_paths = list(image_paths)

		train_dataset = tf.data.Dataset.from_tensor_slices((
			binarization_thresholds[:split_idx_1],
			transcriptions[:split_idx_1],
			image_paths[:split_idx_1]
		))
		valid_dataset = tf.data.Dataset.from_tensor_slices((
			binarization_thresholds[split_idx_1:split_idx_2],
			transcriptions[split_idx_1:split_idx_2],
			image_paths[split_idx_1:split_idx_2]
		))
		test_dataset = tf.data.Dataset.from_tensor_slices((
			binarization_thresholds[split_idx_2:],
			transcriptions[split_idx_2:],
			image_paths[split_idx_2:]
		))

		def process(bin_thresh, transcription, img_path):
			img = tf.io.read_file(img_path)
			img = tf.io.decode_png(img, channels=1)
			img = tf.image.convert_image_dtype(img, tf.float32)
			img = 1 - img
			img = tf.image.resize_with_pad(img, max_h,max_w, method='bilinear')
			img = 1 - img
			if binarize:
				img = tf.where(img > float(bin_thresh), float(1.0), float(0.0))
				img = tf.image.convert_image_dtype(img, tf.uint8)

			# probably should be something else
			label = transcription

			return {"image":img, "label":label}

		train_dataset = train_dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		valid_dataset = valid_dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_dataset = test_dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

		if batch_size:
			assert batch_size > 0
			train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
			valid_dataset = valid_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
			test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

		return train_dataset, valid_dataset, test_dataset
