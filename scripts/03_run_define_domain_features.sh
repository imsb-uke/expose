PYTHONPATH=./ python src/analysis/define_domain_features.py \
    --embedding_dim 768 \
    --expansion_factor 5 \
    --num_classes 1 \
    --domain_classifier_ckpt_path models_domain/2026-08-18_11-49-21/checkpoints/best_model.ckpt \
    --output_path data/domain_weights.pt
