import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



def plot_decision(X, y, clf=None, cm=None):
    '''
    Plot decision function of a given classifier.

    Parameters
    ----------
    X : 2d numpy array
        Array in classical sklearn format (observations by features).
    y : 1d numpy array
        Correct class membership.
    clf : sklearn classifier or Keras model
        Classifier used in predicting class membership.
    cm : colormap
        Colormap to use for class probabilities.
    '''
    assert X.ndim == 2, 'X has to be 2d'

    # if a classifier is supported
    if clf is not None:

        # choose colormap if not given
        if cm is None:
            cm = plt.cm.viridis

        # create a grid of points to check predictions for
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        # check predictions
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        elif hasattr(clf, 'output_layers'):
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # put the result into a contour plot
        Z = Z.reshape(xx.shape)
        cnt = plt.contourf(xx, yy, Z, 12, cmap=cm)
        for c in cnt.collections:
            c.set_edgecolor("face")
        plt.contour(xx, yy, Z, levels=[0.5])

    # create scatterplot for all the observations
    for cls in np.unique(y):
        this_class = y == cls
        plt.scatter(X[this_class, 0], X[this_class, 1],
                    edgecolor='k')

    # add correctness
    if clf is not None:
        pred = clf.predict(X)
        if hasattr(clf, 'output_layers'):
            pred = (pred.ravel() > 0.5).astype('int')
        corr = (pred == y).mean()
        plt.title('correcntess = {}'.format(corr))


def load_images(img_dir, n_images=1000, resize=(50, 50)):
    '''
    Load images of cats and dogs and organize into sklearn-like format.
    '''
    try:
        from keras.preprocessing.image import load_img, img_to_array
    except:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

    images = os.listdir(img_dir)
    czy_pies = np.array(['dog' in img for img in images])
    n_per_categ = n_images // 2

    n_stars = 0
    imgs, y = list(), list()
    for flt_idx, flt in enumerate([~czy_pies, czy_pies]):
        sel_images = np.array(images)[flt]
        np.random.shuffle(sel_images)
        for idx in range(n_per_categ):
            full_img_path = os.path.join(img_dir, sel_images[idx])
            imgs.append(img_to_array(load_img(full_img_path,
                                              target_size=resize)))
            y.append(flt_idx)

            # progressbar
            if idx % 20 == 0:
                print('*', end='')
                n_stars += 1
            if n_stars == 50:
                n_stars = 0
                print('')

    y = np.array(y)
    imgs = np.stack(imgs, axis=0)
    return imgs, y


def apply_modifications(model, custom_objects=None):
    """
    Corrected version of apply_modifications keras_vis library.
    (there is a correct version on github but not on pip)
    """
    try:
        from keras.models import load_model
    except:
        from tensorflow.keras.models import load_model

    fname = next(tempfile._get_candidate_names()) + '.h5'
    model_path = os.path.join(tempfile.gettempdir(), fname)
    model.save(model_path)
    new_model = load_model(model_path, custom_objects=custom_objects)
    os.remove(model_path)
    return new_model


def show_rgb_layers(image, style='light', subplots_args=dict()):
    '''
    Show RGB layers of the image on separate axes.

    Parameters
    ----------
    image : numpy 3d array
        Numpy image array of shape (height, width, RGB)
    style : str
        Style for the display of RGB layers.
    subplots_args : dict
        Additional arguments for the subplots call.

    Returns
    -------
    fig : matplotlib Figure
        Figure object.
    '''
    im_shape = image.shape
    assert im_shape[-1] == 3
    assert image.ndim == 3

    if style == 'light':
        cmaps = ['Reds', 'Greens', 'Blues']

    fig, ax = plt.subplots(ncols=3, **subplots_args)
    for layer in range(3):
        if style == 'light':
            ax[layer].imshow(image[..., layer], cmap=cmaps[layer])
        else:
            temp_img = np.zeros(im_shape[:2] + (3,))
            temp_img[..., layer] = image[..., layer]
            ax[layer].imshow(temp_img)
        ax[layer].axis('off')

    return fig


def extract_features(X, model, batch_size=20):
    '''
    Use a trained model to extract features from training examples.

    Parameters
    ----------
    X : numpy array
        Input data for the model.
    model : keras model
        Keras model to use.
    batch_size : int
        Batch size to use when processing input with the model.

    Returns
    -------
    features : numpy array
        Extracted features (values for the last dense layer of the network
        for example).
    '''
    n_stars = 0
    sample_count = X.shape[0]
    model_shape = (shp.value for shp in model.layers[-1].output.shape[:])
    output_shape = (sample_count,) + tuple(shp for shp in model_shape
                                           if shp is not None)
    features = np.zeros(shape=output_shape)

    n_full_bathes = sample_count // batch_size
    for batch_idx in range(n_full_bathes):
        slc = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        features_batch = model.predict(X[slc])
        features[slc] = features_batch

        # progressbar
        print('*', end='')
        n_stars += 1
        if n_stars == 50:
            n_stars = 0
            print('')

    left_out = sample_count - n_full_bathes * batch_size
    if left_out > 0:
        slc = slice(n_full_bathes * batch_size, None)
        features_batch = model.predict(X[slc])
        features[slc] = features_batch

    features = features.reshape((sample_count, -1))
    return features


def show_image_predictions(X, y, model=None, predictions=None):
    '''FIXME : check what it does and clarify docs'''
    if model is not None and not (predictions is not None):
        predictions = model.predict(X)
    if_correct = np.round(predictions).ravel() == y
    incorrect_predictions = np.where(if_correct == 0)[0]

    # FIXME: change the code below:
    # znajdujemy poprawne przewidywania oraz obliczamy pewność
    confidence = np.abs(predictions.ravel() - 0.5) * 2
    correct_predictions = np.where(if_correct)[0]
    confidence_for_correct_predictions = confidence[correct_predictions]

    # znajdujemy poprawne przedidywania z wysoką pewnością
    high_confidence = np.where(confidence_for_correct_predictions > 0.75)[0]
    correct_high_confidence = correct_predictions[high_confidence]

    # wyświetlamy
    fig, ax = plt.subplots(ncols=6, nrows=3, figsize=(14, 8))
    ax = ax.ravel()

    for idx in range(3 * 6):
        img_idx = correct_high_confidence[idx]
        ax[idx].imshow(X_test[img_idx])
        ax[idx].set_title('{:.2f}%'.format(predictions[img_idx, 0] * 100))
        ax[idx].axis('off')


def test_ipyvolume():
    '''Test ipyvolume installation.'''
    import ipyvolume as ipv

    s = 1/2**0.5
    # 4 vertices for the tetrahedron
    x = np.array([1.,  -1, 0,  0])
    y = np.array([0,   0, 1., -1])
    z = np.array([-s, -s, s, s])
    # and 4 surfaces (triangles), where the number refer to the vertex index
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 3, 2)]

    ipv.figure()
    # draw the tetrahedron mesh
    ipv.plot_trisurf(x, y, z, triangles=triangles, color='orange')
    # mark the vertices
    ipv.scatter(x, y, z, marker='sphere', color='blue')
    # set limits and show
    ipv.xyzlim(-2, 2)
    ipv.show()
