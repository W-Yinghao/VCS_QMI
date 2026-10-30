"""Helper for r27x_flickr_gather.py (run in a subprocess with sys.path[0] = the code dir under test, so the code dir's
own modules are used). Loads STATE, the fixed split (NPZ), runs NSTEPS real A1 training steps of model A and saves
the flattened encoder + refiner parameters of A and B. argv: CODE_DIR STATE NPZ DATA_PATH NOISE_FILE OUT NSTEPS"""
import sys, os
import numpy as np
import torch

code, state, npz, dpath, noise, out, nsteps = sys.argv[1:8]
assert os.path.dirname(os.path.abspath(__import__('r27_interface').__file__)) == os.path.abspath(code)
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab

opt = r25_run.parse(['--noise_file', noise, '--output_dir', os.environ.get('TMPDIR', '/tmp'), '--resume', state, '--data_path', dpath])
opt.arm, opt.lam = 'a1_iface', 0.0
R27.install()
r25_run.seed_all(opt.seed)
vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
opt.vocab_size = len(vocab)
caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
z = np.load(npz)
A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
st = R.load_full_state(state, opt, A, B)
A.avg_clean_prob, B.avg_clean_prob = float(z['acpA']), float(z['acpB'])
lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=z['pred_B'], prob=z['prob_B'])
s = R.new_stats()
R27.train(opt, A, B, lt, ut, st['epoch'] + 1, s, max_iters=int(nsteps))
f = lambda m: torch.cat([p.detach().flatten().double().cpu() for p in list(m.img_enc.parameters()) + list(m.txt_enc.parameters())
                         + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters())])
torch.save({'A': f(A), 'B': f(B), 'gather': getattr(I.Ctx, 'gather_mode', 'original_gpu_images'),
            'raw_i2t': s.get('i2t_raw_loss'), 'raw_t2i': s.get('t2i_raw_loss')}, out)
print('ONESTEP_DONE', code, out)
