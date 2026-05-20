<div align="center">
  <h1>Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer</h1>
  
<div align="center">
  <a href="https://arxiv.org/pdf/2602.12286">[Paper]</a>
</div>

</div>

<div align="center">
  <img src="assets/dna-language-frameworks.png" width="700">
  
  <sub> We systematically investigate three DNA-text fusion strategies. (a) The standard modular architecture, adopted by current DNA-text models. (b) SeqCLIP: explict semantic alignment on the gene encoder by contrastively learning on massive DNA-text pairs. (c) Our One Tokenizer: extend the pre-trained LLM's vocabulary with DNA-specific tokens, allowing LLM to process them natively.</sub>
</div>

## News

- [2026.5.8] The code of our baseline method [SeqCLIP](https://github.com/XenithAI/SeqCLIP) is released. 
- The code of One Tokenizer is coming soon!
  
## Overview
This is the official repository of our paper "Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer". 

A central challenge in developing Multimodal Large Language Models (MLLMs) is effectively integrating heterogeneous inputs into a cohesive reasoning engine. Current paradigms predominantly rely on modular architectures that introduce modality-specific encoders and cross-modal fusion mechanisms. However, these designs are fundamentally bottlenecked by a geometric modality gap, forcing the LLM to expend significant computational capacity on geometric reconciliation rather than deep cross-modal reasoning. In this work, we formally characterize this modality gap and theoretically demonstrate that native architectures, specifically those employing a unified vocabulary, intrinsically maintain a zero-gap state across all hidden layers. 
Guided by these theoretical findings, we propose *One Tokenizer*, a native architecture that maps all modalities directly into a shared token space. We empirically validate this framework on a DNA--text multimodal testbed. Our extensive evaluations reveal that by achieving seamless integration within the LLM's native latent space, One Tokenizer consistently outperforms encoder-based modular counterparts, providing a fundamentally superior framework for deep biological reasoning.


## Experiments
We provide the implementation to reproduce the main results in the paper. 


## Citation

If you find this work useful, please cite it as:

```bibtex
@misc{li2026mindgapmoreachieving,
  title={Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer},
  author={Yanan Li and Christina Yi Jin and Yuan Jin and Manli Luo and Tie Xu and Shuai Jiao and Wei He and Qing Zhao},
  year={2026},
  eprint={2602.12286},
  archivePrefix={arXiv},
  primaryClass={q-bio.GN},
  url={https://arxiv.org/abs/2602.12286},
}
```


