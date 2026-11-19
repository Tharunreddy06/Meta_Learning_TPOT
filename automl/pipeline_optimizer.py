from tpot import TPOTClassifier, TPOTRegressor


class PipelineOptimizer:

    def __init__(
        self,
        problem_type,
        generations=None,
        population_size=None,
        max_time_mins=1,
        max_eval_time_mins=0.1,
        random_state=42,
        cv=3,
        early_stop=2,
        n_jobs=1,
        verbose=0,
        search_space="linear",
        preprocessing=False,
    ):
        self.problem_type = problem_type
        self.generations = generations
        self.population_size = population_size
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.random_state = random_state
        self.cv = cv
        self.early_stop = early_stop
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.search_space = search_space
        self.preprocessing = preprocessing

        self.model = None
        self.best_pipeline = None

    @staticmethod
    def resolve_optimization_mode(mode):
        """Map optimization modes to TPOT 1.1.0 parameters that are actually effective."""

        mode_map = {
            "quick": {
                "max_time_mins": 1.0,
                "max_eval_time_mins": 0.1,
                "cv": 3,
                "early_stop": 2,
                "verbose": 0,
            },
            "balanced": {
                "max_time_mins": 5.0,
                "max_eval_time_mins": 0.5,
                "cv": 5,
                "early_stop": 5,
                "verbose": 0,
            },
            "thorough": {
                "max_time_mins": 10.0,
                "max_eval_time_mins": 1.0,
                "cv": 5,
                "early_stop": 10,
                "verbose": 0,
            },
        }

        key = (mode or "quick").lower()

        if key not in mode_map:
            key = "quick"

        return mode_map[key].copy()

    def create_optimizer(self):
        """Create a TPOT optimizer using only parameters supported by the installed TPOT 1.1.0 API."""

        common_params = {
            "search_space": self.search_space,
            "max_time_mins": self.max_time_mins,
            "max_eval_time_mins": self.max_eval_time_mins,
            "cv": self.cv,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "preprocessing": self.preprocessing,
        }

        if self.population_size is not None:
            common_params["population_size"] = self.population_size
        if self.generations is not None:
            common_params["generations"] = self.generations

        if self.early_stop is not None:
            common_params["early_stop"] = self.early_stop

        if self.problem_type == "classification":
            self.model = TPOTClassifier(**common_params)
        elif self.problem_type == "regression":
            self.model = TPOTRegressor(**common_params)
        else:
            raise ValueError(
                "Problem type must be 'classification' or 'regression'."
            )

        return self.model

    def optimize(self, X, y):

        if self.model is None:
            self.create_optimizer()

        print("\n======================================")
        print("       TPOT PIPELINE OPTIMIZATION")
        print("======================================")

        print(f"Problem Type       : {self.problem_type}")
        print(f"Search Space       : {self.search_space}")
        print(f"Max Time           : {self.max_time_mins} min")
        print(f"Max Pipeline Time  : {self.max_eval_time_mins} min")
        print(f"CV Folds           : {self.cv}")
        print(f"Early Stop         : {self.early_stop}")

        print("\nStarting genetic programming...\n")

        self.model.fit(X, y)

        self.best_pipeline = self.model.fitted_pipeline_

        print("\n======================================")
        print("       OPTIMIZATION COMPLETED")
        print("======================================")

        print("\nBest Pipeline:")
        print(self.best_pipeline)

        return self.best_pipeline

    def get_best_pipeline(self):

        if self.best_pipeline is None:
            raise ValueError(
                "No optimized pipeline available. Run optimize() first."
            )

        return self.best_pipeline

    def get_label_encoder(self):
        """Return TPOT's label encoder for classification."""

        if self.problem_type != "classification":
            return None

        if self.model is None:
            raise ValueError("TPOT optimizer has not been created.")

        return getattr(self.model, "label_encoder_", None)