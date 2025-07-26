package main

import (
	"fmt"
	"time"
)

type Config struct {
	DatasetType string `conf:"dataset-type" help:"The type of dataset to ingest (modis or viirs)" default:"modis"`
}

type IngestionConfig struct {
	CollectionName string
	BaseURL        string
	StartYear      int
	EndYear        int
	FileNamePrefix string
}

func GetIngestionConfig(datasetType string) (IngestionConfig, error) {
	switch datasetType {
	case "modis":
		return IngestionConfig{
			CollectionName: "MODIS",
			BaseURL:        "https://agricultural-production-hotspots.ec.europa.eu/data/MO6_FPAR/MODIS",
			StartYear:      2000,
			EndYear:        2023,
			FileNamePrefix: "mt",
		}, nil
	case "viirs":
		return IngestionConfig{
			CollectionName: "VIIRS",
			BaseURL:        "https://agricultural-production-hotspots.ec.europa.eu/data/MO6_FPAR/MVIIRS",
			StartYear:      2018,
			EndYear:        time.Now().Year(),
			FileNamePrefix: "it",
		}, nil
	default:
		return IngestionConfig{}, fmt.Errorf("unknown dataset type: %s", datasetType)
	}
}
