CREATE TABLE "week6_predictions" (
    team VARCHAR   NOT NULL,
    previous_rank INT NOT NULL,
	predicted_rank DECIMAL(5,3) NOT NULL,
	predicted_ordinal_rank INT NOT NULL,
	actual_rank INT NOT NULL
);