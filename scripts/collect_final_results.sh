#!/bin/bash

set -u

LOGDIR="/home/woody/rlvl/rlvl173v/revelio/logs"
REPORT_DIR="$HOME/revelio/report_artifacts"

RESULT_DIR="$REPORT_DIR/result_summaries"
SCREENSHOT_DIR="$REPORT_DIR/screenshot_text"
TABLE_DIR="$REPORT_DIR/tables"
LOG_COPY_DIR="$REPORT_DIR/logs"
JOB_COPY_DIR="$REPORT_DIR/jobs"

mkdir -p \
    "$RESULT_DIR" \
    "$SCREENSHOT_DIR" \
    "$TABLE_DIR" \
    "$LOG_COPY_DIR" \
    "$JOB_COPY_DIR"

CSV="$TABLE_DIR/selected_final_results.csv"
SUMMARY="$RESULT_DIR/all_final_results.txt"

echo "experiment,model,layer,timestep,epoch,test_accuracy,test_accuracy_percent,test_loss,train_accuracy,train_loss,log_file" > "$CSV"

cat > "$SUMMARY" <<'HEADER'
REVELIO FINAL EXPERIMENT RESULTS
================================

Dataset: Oxford-IIIT Pet
Model: Stable Diffusion 1.5
Classifier: Diff-C
Epochs: 30
Batch size: 4
Prompt: empty
Pooling: GAP

HEADER

add_result () {
    local order="$1"
    local experiment="$2"
    local model="$3"
    local layer="$4"
    local timestep="$5"
    local logfile="$6"

    if [ ! -f "$logfile" ]; then
        echo "MISSING FILE: $logfile"
        return
    fi

    local epoch_line
    local epoch
    local test_accuracy
    local test_loss
    local train_accuracy
    local train_loss
    local accuracy_percent
    local saved_path
    local output_file

    epoch_line=$(grep "Epoch 30: Test Loss" "$logfile" | tail -n 1)

    epoch=$(grep -E "^wandb:[[:space:]]+epoch[[:space:]]+" "$logfile" \
        | tail -n 1 | awk '{print $3}')

    test_accuracy=$(grep -E "^wandb:[[:space:]]+test_accuracy[[:space:]]+" "$logfile" \
        | tail -n 1 | awk '{print $3}')

    test_loss=$(grep -E "^wandb:[[:space:]]+test_loss[[:space:]]+" "$logfile" \
        | tail -n 1 | awk '{print $3}')

    train_accuracy=$(grep -E "^wandb:[[:space:]]+train_accuracy[[:space:]]+" "$logfile" \
        | tail -n 1 | awk '{print $3}')

    train_loss=$(grep -E "^wandb:[[:space:]]+train_loss[[:space:]]+" "$logfile" \
        | tail -n 1 | awk '{print $3}')

    saved_path=$(grep "Results saved to" "$logfile" \
        | tail -n 1 \
        | sed 's/.*Results saved to //')

    # Fallback if WandB summary is unavailable
    if [ -z "$epoch" ]; then
        epoch="30"
    fi

    if [ -z "$test_accuracy" ]; then
        test_accuracy=$(echo "$epoch_line" \
            | sed -n 's/.*Test Accuracy: \([0-9.]*\).*/\1/p')
    fi

    if [ -z "$test_loss" ]; then
        test_loss=$(echo "$epoch_line" \
            | sed -n 's/.*Test Loss: \([0-9.]*\),.*/\1/p')
    fi

    if [ -n "$test_accuracy" ]; then
        accuracy_percent=$(awk -v value="$test_accuracy" \
            'BEGIN { printf "%.3f", value * 100 }')
    else
        test_accuracy="NA"
        accuracy_percent="NA"
    fi

    [ -n "$test_loss" ] || test_loss="NA"
    [ -n "$train_accuracy" ] || train_accuracy="NA"
    [ -n "$train_loss" ] || train_loss="NA"
    [ -n "$saved_path" ] || saved_path="Not found"

    echo "${experiment},${model},${layer},${timestep},${epoch},${test_accuracy},${accuracy_percent},${test_loss},${train_accuracy},${train_loss},$(basename "$logfile")" \
        >> "$CSV"

    {
        echo "------------------------------------------------------------"
        echo "$experiment"
        echo "Model: $model"
        echo "Layer: $layer"
        echo "Timestep: $timestep"
        echo "Epoch: $epoch"
        echo "Test accuracy: $test_accuracy"
        echo "Test accuracy (%): $accuracy_percent"
        echo "Test loss: $test_loss"
        echo "Training accuracy: $train_accuracy"
        echo "Training loss: $train_loss"
        echo "Log file: $logfile"
        echo "Saved output: $saved_path"
        echo
    } >> "$SUMMARY"

    output_file="$SCREENSHOT_DIR/result_${order}.txt"

    {
        echo "============================================================"
        echo "Revelio Result: $experiment"
        echo "============================================================"
        echo "Dataset: Oxford-IIIT Pet"
        echo "Model: $model"
        echo "Layer: $layer"
        echo "Timestep: $timestep"
        echo "Epochs: 30"
        echo "Batch size: 4"
        echo
        echo "Final test accuracy: $accuracy_percent%"
        echo "Final test loss: $test_loss"
        echo "Training accuracy: $train_accuracy"
        echo "Training loss: $train_loss"
        echo
        echo "Original log:"
        echo "$logfile"
        echo
        echo "Saved output:"
        echo "$saved_path"
        echo "============================================================"
    } > "$output_file"

    cp "$logfile" "$LOG_COPY_DIR/"

    local stdout_file="${logfile%.err}.out"
    if [ -f "$stdout_file" ]; then
        cp "$stdout_file" "$LOG_COPY_DIR/"
    fi
}

add_result \
    "01_bottleneck_t25" \
    "SD-1.5 bottleneck, t=25" \
    "Stable Diffusion 1.5" \
    "bottleneck:0" \
    "25" \
    "$LOGDIR/v100_sd15_bottleneck0_t25_1696129.err"

add_result \
    "02_upft0_t25" \
    "SD-1.5 up_ft0, t=25" \
    "Stable Diffusion 1.5" \
    "up_ft:0" \
    "25" \
    "$LOGDIR/v100_sd15_upft0_t25_1696128.err"

# This filename says t50, but the saved result path confirms timestep 25.
add_result \
    "03_upft1_t25" \
    "SD-1.5 up_ft1, t=25" \
    "Stable Diffusion 1.5" \
    "up_ft:1" \
    "25" \
    "$LOGDIR/sd15_upft1_t50_1671566.err"

add_result \
    "04_upft2_t25" \
    "SD-1.5 up_ft2, t=25" \
    "Stable Diffusion 1.5" \
    "up_ft:2" \
    "25" \
    "$LOGDIR/sd15_upft2_t25_1668726.err"

add_result \
    "05_upft1_t50" \
    "SD-1.5 up_ft1, t=50" \
    "Stable Diffusion 1.5" \
    "up_ft:1" \
    "50" \
    "$LOGDIR/sd15_upft1_t50_1668729.err"

add_result \
    "06_upft1_t75" \
    "SD-1.5 up_ft1, t=75" \
    "Stable Diffusion 1.5" \
    "up_ft:1" \
    "75" \
    "$LOGDIR/sd15_upft1_t75_1668730.err"

add_result \
    "07_upft1_t100" \
    "SD-1.5 up_ft1, t=100" \
    "Stable Diffusion 1.5" \
    "up_ft:1" \
    "100" \
    "$LOGDIR/sd15_upft1_t100_from_t75_1693487.err"

cp "$HOME/revelio"/job_v100_sd15_*.sh "$JOB_COPY_DIR/" 2>/dev/null || true
cp "$HOME/revelio"/job_sd15_upft1_*.sh "$JOB_COPY_DIR/" 2>/dev/null || true
cp "$HOME/revelio"/job_sd15_upft2_*.sh "$JOB_COPY_DIR/" 2>/dev/null || true

echo "------------------------------------------------------------" >> "$SUMMARY"
echo "CSV file: $CSV" >> "$SUMMARY"

echo
echo "Collection complete."
echo "Summary: $SUMMARY"
echo "CSV:     $CSV"
echo "Screens: $SCREENSHOT_DIR"
