-- Adapted from QuickDBD: https://www.quickdatabasediagrams.com/

CREATE TABLE "rankings" (
    "team" VARCHAR   NOT NULL,
    "week" INT   NOT NULL,
    "rank" INT   NOT NULL,
    "W" INT,
    "L" INT,
    "winning_perc" DECIMAL(4, 3),
    "opp_rank" INT,
    "opp_P5" INT,
    "home" INT,
    "result" VARCHAR(1),
    "points_scored" INT,
    "points_against" INT,
    "margin" INT,
    "next_week_rank" INT,
    "movement" INT,
	PRIMARY KEY(team, week)
);

