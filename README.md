### Generating synthetic customer reviews
Playground project that can use a data set of arbitrary customer reviews (e.g.
Amazon customer reviews, such as this [data set](https://data.world/datafiniti/consumer-reviews-of-amazon-products) on data.world under BY-NC-SA
license by Datafiniti) in order to produce synthetic reviews about a different topic of choice.

I.e., if we consider this real review about Kindle Paperwhite:

> Great for those that just want an e-reader
> I am enjoying it so far. Great for reading. Had the original Fire since 2012. The Fire used to make my eyes hurt if I read too long. Haven't experienced that with the Paperwhite yet.

..when asked to change topic to a **restaurant**, it will employ OpenAI API to return:

> Great for those that just want a meal
> I am enjoying it so far. Great for dining. Had the original restaurant since 2012. The restaurant used to make my stomach hurt if I ate too much. Haven't experienced that with this restaurant yet.

### Installation

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cp .env.sample .env # and configure API key and desired topic
```

### Usage

Have your dataset in place at `./dataset.csv` (required column names are
*dateUpdated*, *reviews.rating*, *reviews.title*, *reviews.text*). Then running

```bash
python generate.py
```

will write `./processed.csv`

Careful, depending on the length of your reviews, it can bill you ~0.01 USD per review.
