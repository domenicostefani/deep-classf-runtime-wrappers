

echo "mkdir ~/bulkInferenceTest" | ssh mind@$1 /bin/bash
scp data/* mind@$1:~/bulkInferenceTest/
scp bulk_inference_linux_aarch64 mind@$1:~/bulkInferenceTest/
scp *.tflite mind@$1:~/bulkInferenceTest/