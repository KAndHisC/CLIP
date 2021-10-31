
CHECKPOINT_DIR="checkpoint"
MODEL_NAME="ViT-B_16.npz"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
if [ ! -d $CHECKPOINT_DIR ]; then
  mkdir -p $CHECKPOINT_DIR
  echo "makedir $CHECKPOINT_DIR"
fi

# if [ ! -f $CURRENT_DIR/$CHECKPOINT_DIR/$MODEL_NAME ]; then
#   cd $CHECKPOINT_DIR
#   wget https://storage.googleapis.com/vit_models/imagenet21k/$MODEL_NAME
#   cd ..
# fi

python train.py --config CLIP