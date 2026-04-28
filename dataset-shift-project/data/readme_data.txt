Dataset : UCI Adult Income (also called Census Income)
Source  : OpenML, dataset ID 1590
URL     : https://www.openml.org/d/1590
Size    : about 48,842 rows and 14 features. Binary target ("<=50K" vs ">50K").


How to obtain the data

Option 1. Automatic download (recommended)

You do not have to download anything by hand.

When you run

    python main.py

the loader inside src/data_loader.py calls

    sklearn.datasets.fetch_openml(data_id=1590, as_frame=True, parser="auto")

scikit learn downloads the dataset on the first call and caches it on
disk (usually under ~/scikit_learn_data/openml/). Every later run uses
the local cache and does not need internet.


Option 2. Manual download (only if the machine has no internet at the
time you want to run the experiments)

1. From any machine with internet, open
       https://www.openml.org/d/1590
   and download the data in CSV format. Save the file as

       adult.csv

2. Copy adult.csv into THIS folder, so the final path is

       dataset-shift-project/data/adult.csv

3. Open src/data_loader.py and replace the fetch_openml(...) call inside
   the try block with a local read, for example

       df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                     "..", "data", "adult.csv"))
       X = df.drop(columns=["class"])
       y = (df["class"] == ">50K").astype(int)


Option 3. Synthetic fallback (automatic)

If neither of the two options above works at runtime, the loader
silently falls back to sklearn.datasets.make_classification with
10,000 samples and the same 76 / 24 class imbalance, so the pipeline
always finishes. The numbers in the result CSV will be a bit different
but the qualitative conclusions stay the same.


What this folder looks like by default

By default this folder is empty. The dataset itself is fetched and
cached by scikit learn somewhere else on disk. The folder exists so
that anyone who chooses Option 2 above knows exactly where to drop
their adult.csv file.


Attribution

Original source : Becker, B., and Kohavi, R. (1996). UCI Machine
                  Learning Repository. Adult Data Set.
License         : public domain (UCI Machine Learning Repository).
