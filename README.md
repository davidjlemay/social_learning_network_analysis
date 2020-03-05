# social_learning_network_analysis
Implementation of a social learning network optimization algorithm

This implementation based on Brinton, C. G., Buccapatnam, S., Zheng, L., Cao, D., Lan, A. S., Wong, F. M. F., … Poor, H. V. (2018). On the Efficiency of Online Social Learning Networks. _IEEE/ACM Transactions on Networking, 26_(5), 2076–2089. https://doi.org/10.1109/TNET.2018.2859325

The algorithm posits social learning in an online learning environment as a function of learners' knowledge seeking and knowledge dissemination tendencies (or the weighted average of users' questions and responses by topics over the duration of an online course). Finding optimal the network connecting knowledge-seekers and knowledge-disseminators is cast as a convex optimization problem and is solved by a derived projected gradient descent approach employing the alternating direction method of multipliers. The algorithm offers a means to evaluate the effectiveness of a social learning network and can be used to encourage more optimal interactions between users.
