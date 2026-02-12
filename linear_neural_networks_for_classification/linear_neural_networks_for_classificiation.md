# Linear neural networks for classificiation

Resume at: 4.5 Concise implementation of Softmax Regression

Learning with Mistral as a tutor if I need help:
> https://chat.mistral.ai/chat/f4fe68dc-623c-47cf-b116-a17f6803a724

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
