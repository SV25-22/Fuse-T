# Time thresholds in minutes
MINUTES=(10 30 60 120 240)
EVENTS=("charliehebdo-all-rnr-threads" "ebola-essien-all-rnr-threads" "ferguson-all-rnr-threads" "germanwings-crash-all-rnr-threads" "gurlitt-all-rnr-threads" "ottawashooting-all-rnr-threads" "prince-toronto-all-rnr-threads" "putinmissing-all-rnr-threads" "sydneysiege-all-rnr-threads")

for MINS in "${MINUTES[@]}"; do
    for EVENT in "${EVENTS[@]}"; do
        echo "=================================================="
        echo "Running Early Detection: ${MINS} mins | Fold: $EVENT"
        echo "=================================================="
        
        # 1. GNN-only at T minutes
        python src/train_gnn.py \
            --fold "$EVENT" \
            --early_minutes "$MINS" \
            --out_dir "results/early_${MINS}/gnn/$EVENT"
            
        # 2. Fuse-T at T minutes
        # Re-using the Text checkpoint from Run 1, as text representation doesn't change with graph propagation time
        python src/train_fusion_gem.py \
            --fold "$EVENT" \
            --early_minutes "$MINS" \
            --out_dir "results/early_${MINS}/fuse_t/$EVENT" \
            --init_text_ckpt "results/text/$EVENT/best.pt" \
            --init_gnn_ckpt "results/early_${MINS}/gnn/$EVENT/best.pt"
    done
done