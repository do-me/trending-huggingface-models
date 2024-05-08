# Trending HuggingFace Models
Notifications and ready-to-use exports (csv, xlsx, parquet, json, html) with trending feature-extraction models for downstream applications using transformers.js. 

The data is mined daily from https://huggingface.co/models?library=transformers.js&other=feature-extraction&sort=trending and the individual model's pages (for onnx file size).

Sends notifications to 3 channels in ntfy: 
- Daily updates:
- Weekly updates:
- Monthly updates:

Originally desgined for [SemanticFinder](https://github.com/do-me/SemanticFinder) but has potential for other use cases.

PR's highly appreciated! 

## To Dos
- Add rank delta to see what models are becoming more or less popular
- Add MTEB scores
- Allow for more channels with different ranking (like, downloads, trending, MTEB scores...), whatever might suit your needs
- Add other models, not only for feature-extraction
