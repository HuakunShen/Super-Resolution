# Dataset Setup

We used 2 Datasets

- DIV2K
- TEXT

We made some modifications to the Original Datasets and generated our own.

## Setup

```bash
# run the bash script within this directory, it will download all datasets and unzip them inplace.
bash setup.sh
```

> At least 15GB of space is expected for all datasets.

The `setup.sh` uses `wget` to download datasets from a raspberry pi server running at my home, so it might not be very stable. If it doesn't work, you can always use the cloud drive links below to download the datasets and `unzip` them manually.

[DIV2KCustom Download Link](https://1drv.ms/u/s!AtWR2LUs_Xh6ie8QVcZzsVhygcqlbQ?e=4h7DUG)

[TEXT Download Link](https://1drv.ms/u/s!AtWR2LUs_Xh6ie8dD0p8AeGGllQTFg?e=xtnAxG)

After downloading the datasets, unzip them in such pattern in order to use them for training and testing.

```
Super-Resolution
└── datasets
    ├── DIV2K
    │   ├── diff
    │   ├── same
    │   ├── same_300
    └── TEXT
        ├── diff
        └── same
```



See each directory for more details:

- [DIV2KCustom](./DIV2K)
- [TEXT](./TEXT)

