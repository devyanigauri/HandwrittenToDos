import os


class WordsLoader:
	def __init__(self, datapath=os.path.normpath('../data/'), verbose=False)
		self.datapath = datapath
		self.words_dir_path = os.path.join(self.datapath,'words')
		self.words_txt_path = os.path.join(self.datapath,'ascii/words.txt')

		self.attributes = ['id','clean','binarizationThreshold','x','y','w','h','tag','transcription','path']

		assert os.path.isdir(self.datapath), 'need /data at toplevel of repo'
		assert os.path.isdir(self.words_dir_path), 'need /words directory in /data directory'
		assert os.path.isfile(self.words_txt_path), 'need /ascii/words.txt file in /data directory'

		self.data = {}

		with open(self.words_txt_path, mode='r') as f:
			if verbose:
				print(f'Reading file {self.words_txt_path}...',end='')
			raw = f.read()
			if verbose:
				print(' done!')

		if verbose:
			print('Processing words.txt info...',end='')

		for line in raw.split('\n'):
			if (not line[0] == '#') and line:
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

		if verbose:
			print(f"Finding all PNG files in {self.words_dir_path}...")

		image_paths = glob.glob(self.words_dir_path + '/**/*.png', recursive=True)

		# add image paths to self.data

		self._unfiltered_data = self.data.copy()

		if verbose:
			print(f' done! {len(self.data)} entries indexed.')

	def clear_filters(self):
		self.data = self._unfiltered_data.copy()

	def filter(self, attribute, condition):
		new_data = {}
		for k in self.data.keys():
			if condition(self.data[k][attribute]):
				new_data[k] = self.data[k].copy()
		self.data = new_data

	def load(self, verbose=False):

 
