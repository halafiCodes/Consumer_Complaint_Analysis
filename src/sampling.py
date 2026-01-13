class StratifiedComplaintSampler:
    """
    Stratified sampler for CFPB complaints.

    Ensures balanced representation across product categories
    when creating a reduced-size dataset for embedding and
    vector store construction.
    """

    def __init__(self, target_size: int, random_state: int = 42):
        self.target_size = target_size
        self.random_state = random_state

    def sample(self, df: pd.DataFrame, product_col: str = "Product") -> pd.DataFrame:
        products = df[product_col].unique()
        samples_per_product = self.target_size // len(products)

        sampled_dfs = []
        for product in products:
            group = df[df[product_col] == product]
            n = min(len(group), samples_per_product)
            sampled_dfs.append(
                group.sample(n=n, random_state=self.random_state)
            )

        return pd.concat(sampled_dfs).reset_index(drop=True)
