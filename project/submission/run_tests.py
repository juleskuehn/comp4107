import Embed_Models as embeddings_from_titles
import Embed_Models_GloVe as embeddings_from_glove

test_run = embeddings_from_titles.main(
    epochs=5, lr=0.01, embed_size=32, batch_size=512, mlp=False, cnn=False)

# model1 = embeddings_from_titles.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=False, cnn=False)
# model2 = embeddings_from_titles.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=True, cnn=False)
# model3 = embeddings_from_titles.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=False, cnn=True)

# model4 = embeddings_from_glove.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=False, cnn=False,
#     trainable=False)
# model5 = embeddings_from_glove.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=True, cnn=False,
#     trainable=False)
# model6 = embeddings_from_glove.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=False, cnn=True,
#     trainable=False)

# model7 = embeddings_from_glove.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=False, cnn=False,
#     trainable=True)
# model8 = embeddings_from_glove.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=True, cnn=False,
#     trainable=True)
# model9 = embeddings_from_glove.main(
#     epochs=25, lr=0.001, embed_size=32, batch_size=128, mlp=False, cnn=True,
#     trainable=True)