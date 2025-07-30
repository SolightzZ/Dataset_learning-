# Naive Bayes

# p(spwm) = type Spam 5/10 0.50 = p(A)

# Free = Y -- Y Y Y Spam = S = 3/5 = 0.60
# Free = N -- N Y Y Spam = S = 2/5 = 1.00
# Free - Y -- N Y Y Spam = S = 2/5 = 0.40




# TEST 

# F/A       TN      TS
# Free Won Cash Type
# ?     ?   ?     ?

# P(Ci | X) = p(x|ci) p(Ci)
# P(Type=normal) = | Xtest11 = 0 ค่ามากสุดแสดง ว่าคือ Class อะไร? ของ Spam
# P(Type=normal) Xtest11 = 0.6 
#                   |
#                P(N=Y) 0.6
#                P(N=N) 0.4


#       ------------ TEST 12 --------------

# P (normal\x12 = P(Free=y\Normal P(Won=y\Normal P(Cash=y\Normal))
# = (0.00 x 1.00 x 0.00 x 0.5) = 0
# P (Spam\x12 = P(Free=y\Spam P(Won=n\Spam P(Cash=y\Spam))
# = (0.60 x 0.40 x 0.60 x 0.50 ) = 0.048 
