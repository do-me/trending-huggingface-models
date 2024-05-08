# Trending HuggingFace Models
Notifications and ready-to-use exports (csv, xlsx, parquet, json, html) with trending feature-extraction models for downstream applications using transformers.js. 

The data is mined daily from https://huggingface.co/models?library=transformers.js&other=feature-extraction&sort=trending and the individual model's pages (for onnx file size).

![image](https://github.com/do-me/trending-huggingface-models/assets/47481567/498e3c65-2def-41a8-9022-4803f5d9be7e)

Sends notifications to 3 channels in ntfy: 
- Daily updates: https://ntfy.sh/feature_extraction_transformers_js_models_daily
- Weekly updates: https://ntfy.sh/feature_extraction_transformers_js_models_weekly
- Monthly updates: https://ntfy.sh/feature_extraction_transformers_js_models_monthly

Originally desgined for [SemanticFinder](https://github.com/do-me/SemanticFinder) but has potential for other use cases.

PR's highly appreciated! 

## To Dos
- Add rank delta to see what models are becoming more or less popular
- Add caching to download only new meta data
- Add MTEB scores
- Allow for more channels with different ranking (like, downloads, trending, MTEB scores...), whatever might suit your needs
- Add other models, not only for feature-extraction
