package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/paulmach/orb"
	"github.com/paulmach/orb/encoding/wkt"
	pb "github.com/tilebox/dclimate-labs-vci-ingestion/protogen/tilebox/v1"
	"github.com/tilebox/tilebox-go/datasets/v1"
	ds "github.com/tilebox/tilebox-go/protogen/datasets/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const (
	datasetName    = "tilebox.modis_fpar"
	collectionName = "MODIS"
	baseURL        = "https://agricultural-production-hotspots.ec.europa.eu/data/MO6_FPAR/MODIS"
	startYear      = 2000
	endYear        = 2023
	dekads         = 36
)

func main() {
	ctx := context.Background()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	client := datasets.NewClient()
	dataset, err := client.Datasets.Get(ctx, datasetName)
	if err != nil {
		slog.Error("failed to get dataset", "error", err)
		return
	}

	collection, err := client.Collections.GetOrCreate(ctx, dataset.ID, collectionName)
	if err != nil {
		slog.Error("failed to get or create collection", "error", err)
		return
	}

	for year := startYear; year <= endYear; year++ {
		for dekad := 1; dekad <= dekads; dekad++ {
			fileName := fmt.Sprintf("mt%02d%02dFPRCF.txt", year%100, dekad)
			url := fmt.Sprintf("%s/%d/CF/%s", baseURL, year, fileName)

			slog.Info("processing file", "url", url)

			req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
			if err != nil {
				slog.Error("failed to create request", "url", url, "error", err)
				return
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				slog.Error("failed to fetch file", "url", url, "error", err)
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				slog.Warn("file not found or accessible", "url", url, "status", resp.Status)
				continue
			}

			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				slog.Error("failed to read response body", "url", url, "error", err)
				return
			}

			metadata, err := parseMetadata(string(bodyBytes))
			if err != nil {
				slog.Error("failed to parse metadata", "url", url, "error", err)
				return
			}

			datapoint, err := mapToDatapoint(metadata, url)
			if err != nil {
				slog.Error("failed to map metadata to datapoint", "url", url, "error", err)
				return
			}

			datapoints := []*pb.ModisFpar{datapoint}
			_, err = client.Datapoints.Ingest(ctx, collection.ID, &datapoints, true)
			if err != nil {
				slog.Error("failed to ingest datapoint", "url", url, "error", err)
				return
			}
			slog.Info("successfully ingested datapoint", "url", url)
		}
	}
}

func parseMetadata(body string) (map[string]string, error) {
	metadata := make(map[string]string)
	scanner := bufio.NewScanner(strings.NewReader(body))

	inMetadataSection := false
	re := regexp.MustCompile(`\s*([^=]+?)\s*=\s*(.*)\s*`)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "Metadata:") {
			inMetadataSection = true
			continue
		}
		if strings.HasPrefix(line, "Image Structure Metadata:") {
			inMetadataSection = false
			continue
		}

		if inMetadataSection {
			matches := re.FindStringSubmatch(line)
			if len(matches) == 3 {
				metadata[strings.TrimSpace(matches[1])] = strings.TrimSpace(matches[2])
			}
		}
	}

	// Size
	re = regexp.MustCompile(`Size is (\d+), (\d+)`)
	matches := re.FindStringSubmatch(body)
	if len(matches) > 2 {
		metadata["width"] = matches[1]
		metadata["height"] = matches[2]
	}

	// Pixel Size
	re = regexp.MustCompile(`Pixel Size = \(([^,]+),([^)]+)\)`)
	matches = re.FindStringSubmatch(body)
	if len(matches) > 2 {
		metadata["pixel_size_x"] = matches[1]
		metadata["pixel_size_y"] = matches[2]
	}

	// CRS from AUTHORITY
	re = regexp.MustCompile(`AUTHORITY\["EPSG","(\d+)"\]`)
	allMatches := re.FindAllStringSubmatch(body, -1)
	if len(allMatches) > 0 {
		lastMatch := allMatches[len(allMatches)-1]
		if len(lastMatch) > 1 {
			metadata["crs"] = "EPSG:" + lastMatch[1]
		}
	}

	// Corner Coordinates for Polygon
	re = regexp.MustCompile(`Upper Left\s+\(([^,]+),\s+([^)]+)\)`)
	ul := re.FindStringSubmatch(body)
	re = regexp.MustCompile(`Lower Left\s+\(([^,]+),\s+([^)]+)\)`)
	ll := re.FindStringSubmatch(body)
	re = regexp.MustCompile(`Upper Right\s+\(([^,]+),\s+([^)]+)\)`)
	ur := re.FindStringSubmatch(body)
	re = regexp.MustCompile(`Lower Right\s+\(([^,]+),\s+([^)]+)\)`)
	lr := re.FindStringSubmatch(body)

	if len(ul) > 2 && len(ll) > 2 && len(ur) > 2 && len(lr) > 2 {
		metadata["upper_left"] = fmt.Sprintf("%s %s", ul[1], ul[2])
		metadata["lower_left"] = fmt.Sprintf("%s %s", ll[1], ll[2])
		metadata["upper_right"] = fmt.Sprintf("%s %s", ur[1], ur[2])
		metadata["lower_right"] = fmt.Sprintf("%s %s", lr[1], lr[2])
	}

	return metadata, nil
}

func mapToDatapoint(metadata map[string]string, url string) (*pb.ModisFpar, error) {
	p := &pb.ModisFpar{}
	assetURL := strings.Replace(url, ".txt", ".tif", 1)
	p.SetAssetUrl(assetURL)
	p.SetCreator(metadata["creator"])
	p.SetSensorType(metadata["sensor_type"])
	p.SetConsolidationPeriod(metadata["consolidation_period"])
	p.SetLineage(metadata["lineage"])
	p.SetProgramVersion(metadata["program"])
	p.SetCrs(metadata["crs"])

	// Time fields
	if val, ok := metadata["date"]; ok {
		t, err := time.Parse("20060102", val)
		if err == nil {
			p.SetTime(timestamppb.New(t))
		} else {
			return nil, fmt.Errorf("failed to parse date: %w", err)
		}
	} else {
		return nil, fmt.Errorf("metadata is missing 'date' field")
	}

	if val, ok := metadata["file_creation"]; ok {
		t, err := time.Parse("2006-01-02T15:04:05", val)
		if err == nil {
			p.SetFileCreation(timestamppb.New(t))
		}
	}

	// Numeric fields
	if val, ok := metadata["days"]; ok {
		if i, err := strconv.ParseInt(val, 10, 64); err == nil {
			p.SetDays(i)
		}
	}
	if val, ok := metadata["width"]; ok {
		if i, err := strconv.ParseInt(val, 10, 64); err == nil {
			p.SetWidth(i)
		}
	}
	if val, ok := metadata["height"]; ok {
		if i, err := strconv.ParseInt(val, 10, 64); err == nil {
			p.SetHeight(i)
		}
	}
	if val, ok := metadata["pixel_size_x"]; ok {
		if f, err := strconv.ParseFloat(strings.TrimSpace(val), 64); err == nil {
			p.SetPixelSizeX(f)
		}
	}
	if val, ok := metadata["pixel_size_y"]; ok {
		if f, err := strconv.ParseFloat(strings.TrimSpace(val), 64); err == nil {
			p.SetPixelSizeY(f)
		}
	}
	if val, ok := metadata["lowest_actual_value"]; ok {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			p.SetLowestActualValue(f)
		}
	}
	if val, ok := metadata["highest_actual_value"]; ok {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			p.SetHighestActualValue(f)
		}
	}
	if val, ok := metadata["lowest_possible_value"]; ok {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			p.SetLowestPossibleValue(f)
		}
	}
	if val, ok := metadata["highest_possible_value"]; ok {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			p.SetHighestPossibleValue(f)
		}
	}

	// Array fields
	if val, ok := metadata["flags"]; ok {
		p.SetFlags(strings.Split(val, ", "))
	}

	// Geometry
	if ul, ok := metadata["upper_left"]; ok {
		if ll, ok := metadata["lower_left"]; ok {
			if lr, ok := metadata["lower_right"]; ok {
				if ur, ok := metadata["upper_right"]; ok {
					polyStr := fmt.Sprintf("POLYGON((%s, %s, %s, %s, %s))",
						strings.TrimSpace(ul),
						strings.TrimSpace(ll),
						strings.TrimSpace(lr),
						strings.TrimSpace(ur),
						strings.TrimSpace(ul))
					g, err := wkt.Unmarshal(polyStr)
					if err == nil {
						geom := ds.NewGeometry(g.(orb.Polygon))
						p.SetGeometry(geom)
					} else {
						slog.Warn("failed to unmarshal WKT polygon", "wkt", polyStr, "error", err)
					}
				}
			}
		}
	}

	return p, nil
}
