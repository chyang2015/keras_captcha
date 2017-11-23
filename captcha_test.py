from captcha_model import captcha_model
from captcha_generate_image import gen,decode
import matplotlib.pyplot as plt
import pylab
model = captcha_model()
model.load_weights("net-weight\\net-epoch.hdf5")
X,y = next(gen(1))
y_pred = model.predict(X)
plt.title('real:%s\npred:%s' % (decode(y),decode(y_pred)))
plt.imshow(X[0].transpose((1,2,0)),cmap='gray')
pylab.show()