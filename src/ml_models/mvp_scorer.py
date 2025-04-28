from src.models.mvp_basic_data import BasicProfile, DeclaredIncome

def calculate_basic_score(profile: BasicProfile, income: DeclaredIncome) -> float:
    """
    Calculates a basic credit score for the MVP based on declared income and profile data.
    This is a simplified logic for initial testing.
    """
    # Simple scoring logic: Higher income and more years in business lead to a higher score.
    # This is a placeholder and should be replaced with more sophisticated logic later.
    score = 0.0

    if income and income.monthly_income is not None:
        # Scale income to contribute to the score. Example: 1000 income adds 10 points.
        score += income.monthly_income / 100.0

    if profile and profile.years_in_business is not None:
        # Add points based on years in business. Example: 1 year adds 5 points.
        score += profile.years_in_business * 5.0

    # Ensure a minimum score or handle cases with no data if necessary
    # For this MVP, we'll just return the calculated score.
    return score

# Note: This function assumes BasicProfile and DeclaredIncome objects are passed.
# In a real scenario, you might pass dictionaries or Pydantic models and handle validation.
