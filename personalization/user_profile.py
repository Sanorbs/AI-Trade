class UserProfile:
    def __init__(self, risk_tolerance='balanced', target_return=0.1, feedback_weight=0.5):
        self.risk_tolerance = risk_tolerance  # 'conservative', 'balanced', 'aggressive'
        self.target_return = target_return
        self.feedback_weight = feedback_weight

    def set_risk_tolerance(self, risk_tolerance):
        self.risk_tolerance = risk_tolerance

    def set_target_return(self, target_return):
        self.target_return = target_return

    def set_feedback_weight(self, feedback_weight):
        self.feedback_weight = feedback_weight

    def get_profile(self):
        return {
            'risk_tolerance': self.risk_tolerance,
            'target_return': self.target_return,
            'feedback_weight': self.feedback_weight
        } 