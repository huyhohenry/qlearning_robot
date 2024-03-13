

import random as rand

import numpy as np


class QLearner(object):
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    This is a Q learner object.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param num_states: The number of states to consider.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type num_states: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param num_actions: The number of actions available..  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type num_actions: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type alpha: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type gamma: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type rar: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type radr: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type dyna: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type verbose: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Constructor method  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        # Q table is a 2D matrix: num_states x num_actions. For each state, their will be num_actions of action.
        # Q table stores the Q value of each action within each states
        # Q table initialized to 0
        self.Q = np.zeros(shape=(num_states, num_actions))
        # R matrix used in dyna. To store rewards from real life interaction
        self.R = np.zeros(shape=(num_states, num_actions))
        # T dictionary used in dyna. Store counter of moving to state s_prime when in state s taking action a.
        # This is Tc in lecture. No need to convert to probability since picking the max only.
        self.T = {}
        # Experience list
        self.exp = []

    def author(self):
        return ""

    def querysetstate(self, s):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Update the state without updating the Q-table  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param s: The new state  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type s: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: The selected action  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # Logic to determine the next action
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, 3)
        else:
            action = self.Q[s, :].argmax()
        # Keep track of new state and action
        self.a = action
        self.s = s
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Update the Q table and return an action  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param s_prime: The new state  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type s_prime: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param r: The immediate reward  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type r: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: The selected action  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # Given the new state s_prime and immediate reward for moving into s_prime, update Q tables
        self.Q[self.s, self.a] = (1-self.alpha)*self.Q[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * self.Q[s_prime, self.Q[s_prime,:].argmax()])
        # Append to experience list
        self.exp.append((self.s, self.a, s_prime, r))
        # Implement dyna
        if self.dyna > 0:
            # Update Reward matrix:
            self.R[self.s, self.a] = (1-self.alpha)*self.R[self.s, self.a] + self.alpha * r
            # Update Transition matrix
            if (self.s, self.a) not in self.T:
                self.T[(self.s, self.a)] = {s_prime: 1}
            else:
                if s_prime not in self.T[(self.s, self.a)]:
                    self.T[(self.s, self.a)][s_prime] = 1
                else:
                    self.T[(self.s, self.a)][s_prime] += 1
        # Create a random list size of dyna, of integers from index of experience list.
        exp_random_ind = np.random.randint(len(self.exp), size=self.dyna)
        for i in range(self.dyna):
            # pick a random state and action that are in the experience
            s, a, s_prime_no_use, r_no_use = self.exp[exp_random_ind[i]]
            # Infer S-prime from T matrix
            all_possible_states = self.T[(s, a)]
            s_prime_inf = max(all_possible_states, key = all_possible_states.get)
            # Update Q table using s, a, s_prime, r
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] \
                      + self.alpha * (self.R[s, a] + self.gamma * self.Q[s_prime_inf, self.Q[s_prime_inf, :].argmax()])

        # Logic to determine the next action
        if rand.uniform(0.0, 1.0) < self.rar:
            action = np.random.randint(0,3)
        else:
            action = self.Q[s_prime, :].argmax()

        # Update rar
        self.a = action
        self.s = s_prime
        self.rar *= self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")

