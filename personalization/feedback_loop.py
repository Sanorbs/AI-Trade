class FeedbackLoop:
    def __init__(self, agent, user_profile):
        self.agent = agent
        self.user_profile = user_profile

    def update_agent(self, trading_result):
        # Example: adapt exploration rate or risk based on user profile and results
        if self.user_profile.risk_tolerance == 'conservative':
            self.agent.epsilon = max(self.agent.epsilon, 0.2)
        elif self.user_profile.risk_tolerance == 'aggressive':
            self.agent.epsilon = min(self.agent.epsilon, 0.8)
        # Adapt based on trading result (e.g., if loss, become more conservative)
        if trading_result < 0:
            self.agent.epsilon = min(self.agent.epsilon + 0.05, 1.0)
        else:
            self.agent.epsilon = max(self.agent.epsilon - 0.05, 0.01)
        # Optionally, adapt other agent parameters here

    def get_agent_params(self):
        return {
            'epsilon': self.agent.epsilon
        } 