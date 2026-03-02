# Linear neural networks for classificiation

Resume at: 4.7.3.2. Covariate Shift Correction

Learning with Mistral as a tutor if I need help:
> https://chat.mistral.ai/chat/f4fe68dc-623c-47cf-b116-a17f6803a724
> https://chat.mistral.ai/chat/1d77e5f8-c5d7-408f-b034-07ed780bb4a7

## 4.3.4 Exercises

1. m = taille totale de l'ensemble de validation ; k = taille d'un minibatch ; n = m/k est le nombre de minibatches
L_v = (1/m)*([l_i for i in range(m)]) où l_i est la perte pour le i-ème exemple.

2. L^q_v = moyenne des pertes sur les minibatches

3. -

## 4.4.7 Exercises

1. 1. Lorsqu'on utilise softmax avecv 100 en input, le problème c'est que calculer l'exponentielle sera super long. Genre là exp(100).....
   2. Ok avec des valeurs super petites, le problème c'est que c'est ULTRA proche de 0, donc le PC va juste arrondir à 0 et on perd des infos.
   3. Pour éviter overflow/underflow : softmax(xi​)=∑j​exp(xj​)exp(xi​)​=∑j​exp(xj​−c)exp(xi​−c)​. En python ça donne :
    ```
    def stable_softmax(X):
      X_max=tf.reduce_max(X,axis=1,keepdim=True)
      X_exp=tf.exp(X-X_max)
      partition=tf.reduce_sum(X_exp,axis=1,keepdim=True)
      return X_exp/partition
    ```

2. 1. Fonction Cross-Entropy :
    ```
    def cross_entropy(y_hat,y):
      log_probs = tf.math.log(tf.boolean_mask(y_hat,tf.one_hot(y, depth=y_hat.shape[-1])))
      return -tf.reduce_mean(log_probs)
    ```
   2. C'est sûrement plus lent à cause de tf.boolean_mask & tf.one_hot ajoutent des opérations supplémentaires.
   3. Utilisable soit pour comprendre comment C-E fonctionne, soit pour petits jeux de données où osef de la perf.
   4. A cause du logarithme, faire attention au log où y_hat[i] = 0

3. Ce sont des probabilités donc non, si on a 51% qui est le max, 49% indiquent que ça peut être mauvais aussi... Pour fix, on peut utiliser un seuil, ...

4. Imagine vocab de 50 000 mots, donc exp(o[i]) calculé chaque fois c'est beaucoup. On peut organiser le vocab en arbre, ...

5. Un taux trop petit → Convergence lente.
Un taux trop grand → Divergence (la perte explose).
Un taux optimal → Convergence rapide et stable.

## 4.5.5 Exercises

1. Normalement underflow arrive si exp(x) devient trop petit pour être représenté, et à l'inverse exp(x) trop grand...

```python
print(np.log(np.finfo(np.float64).max))
print(np.log(np.finfo(np.float64).tiny))
#etc
```

2. log_2(x), quantification affine (x_quant = round(x/s) + z, avec s un facteur d'échelle et z un zéro-point)

3. Overfitting (modèle connait / coeur nos données de train). On peut fix en rajoutant des données, ou faire de l'early stopping (genre si le modèle détecte 10x d'affilée que l'on ne fait pas de progrès loss / validation alors stop).

4. Trop petit = modèle apprend trop lentement.
Trop grand = la loss oscille ou est trop grande. J'avais utilié du grid search pour tester ça.

## 4.6.1 The Test Set

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="ORD">
    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">O</mi>
  </mrow>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mrow data-mjx-texclass="ORD">
    <mo>/</mo>
  </mrow>
  <msqrt>
    <mi>n</mi>
  </msqrt>
  <mo stretchy="false">)</mo>
</math>

Erreur réelle :
E(x,y)∼P​[1(f(x)!=y)]

"""
With these misgivings in mind, you might now be sufficiently primed to see the appeal of statistical learning theory, the mathematical subfield of machine learning whose practitioners aim to elucidate the fundamental principles that explain why/when models trained on empirical data can/will generalize to unseen data. One of the primary aims of statistical learning researchers has been to bound the generalization gap, relating the properties of the model class to the number of samples in the dataset.
"""

Donc on utilise les stats en ML pour comprendre pourquoi mon modèle généralise sur mes données, ou pourquoi il ne généralise pas.

- Question : est-ce qu'il ne faudrait pas demander à quelqu'un d'extérieur de récupérer un set de test pour nous ? Comme ça ce n'est pas biaisé ? Je suis sûrement hors-sujet là. Mais comme ça, ce ne serait pas biaisé.

## 4.6.5 Exercises

1. Sans même calculer, selon les ordres de grandeur indiqués auparavant, au moins des millions (voire milliard ?)
2. m*(k-1) + 1 avec m étant le nombre d'échantillons du test set, et k le nombre de classes possibles
3. -
4. -

