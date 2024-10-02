package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/google/go-github/github"
	openai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
	"golang.org/x/oauth2"
	"gopkg.in/yaml.v3"
)

type Config struct {
	RepoNames   []string `yaml:"repo_names"`
	UserNames   []string `yaml:"user_names"`
	OpenAIToken string   `yaml:"openai_token"`
}

// PREvent defines an enum
type PREvent string

const (
	APPROVE         PREvent = "APPROVE"
	REQUEST_CHANGES PREvent = "REQUEST_CHANGES"
	COMMENT         PREvent = "COMMENT"
)

type Response struct {
	Status  PREvent `json:"status"`
	Message string  `json:"message"`
}

type OpenAIResponse struct {
	Qs []struct {
		Q string `json:"q"`
		G string `json:"g"`
	}
}

type PullRequestReviewRequest struct {
	Body  string  `json:"body,omitempty"`
	Event PREvent `json:"event,omitempty"`
}

var config Config

func main() {
	// load config
	configFile, err := os.ReadFile("config.yml")
	if err != nil {
		log.Fatalf("failed to open config file: %v", err)
	}

	err = yaml.Unmarshal(configFile, &config)
	if err != nil {
		log.Fatalf("failed to unmarshal config file: %v", err)
	}

	// config routes
	http.HandleFunc("/checkrepo", checkRepoHandler)
	http.HandleFunc("/", staticHandler)

	log.Print("Server started on port 8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func checkRepoHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	repoURL := r.URL.Query().Get("repo")
	if repoURL == "" {
		http.Error(w, "repo query parameter is required", http.StatusBadRequest)
		return
	}

	ghToken := r.URL.Query().Get("token")
	if ghToken == "" {
		http.Error(w, "token query parameter is required", http.StatusBadRequest)
		return
	}

	prNumStr := r.URL.Query().Get("pr")
	if prNumStr == "" {
		http.Error(w, "pr query parameter is required", http.StatusBadRequest)
		return
	}
	prNum, err := strconv.Atoi(prNumStr)
	if err != nil {
		http.Error(w, "pr query parameter must be an integer", http.StatusBadRequest)
		return
	}

	// validate and parse github repo url
	userName, repoName, err := ParseGitHubRepoURL(repoURL)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// check if user is allowed
	if !slices.Contains(config.UserNames, userName) {
		http.Error(w, "user not allowed", http.StatusForbidden)
		return
	}

	// check if repo is allowed
	if !slices.Contains(config.RepoNames, repoName) {
		http.Error(w, "repo not allowed", http.StatusForbidden)
		return
	}

	// check if repo exists
	ctx := r.Context()
	client := github.NewClient(oauth2.NewClient(ctx, oauth2.StaticTokenSource(&oauth2.Token{AccessToken: ghToken})))
	_, ghStatus, err := client.Repositories.Get(ctx, userName, repoName)
	if (err != nil) || (ghStatus.StatusCode != http.StatusOK) {
		http.Error(w, "error retrieving repo from github api", http.StatusBadRequest)
		return
	}

	// check if pr exists in repo
	ghPr, ghStatus, err := client.PullRequests.Get(ctx, userName, repoName, prNum)
	if (err != nil) || (ghStatus.StatusCode != http.StatusOK) {
		http.Error(w, "error retrieving pr from github api", http.StatusBadRequest)
		return
	}

	// check if ghToken is a valid actions token for this repo by hitting the /installation/repositories endpoint
	// _, ghStatus, err = client.Apps.ListRepos(ctx, &github.ListOptions{})
	// if (err != nil) || (ghStatus.StatusCode != http.StatusOK) {
	// 	http.Error(w, "error validating token", http.StatusUnauthorized)
	// 	return
	// }

	// Get the head branch name from the pull request
	branchName := ghPr.GetHead().GetRef()
	if branchName == "" {
		http.Error(w, "Error getting head branch from pull request", http.StatusInternalServerError)
		return
	}

	// Retrieve the /README.md file from the repo
	fileContent, _, _, err := client.Repositories.GetContents(ctx, userName, repoName, "README.md", &github.RepositoryContentGetOptions{
		Ref: branchName,
	})
	if err != nil {
		response := Response{
			Status:  "REQUEST_CHANGES",
			Message: "Error retrieving README.md file.",
		}
		createReview(ctx, ghToken, userName, repoName, prNum, response)
		writeHTTPResponse(w, response)
		return
	}

	// Decode the content of the README.md file
	content, err := fileContent.GetContent()
	if err != nil {
		response := Response{
			Status:  "REQUEST_CHANGES",
			Message: "Error retrieving README.md file.",
		}
		createReview(ctx, ghToken, userName, repoName, prNum, response)
		writeHTTPResponse(w, response)
		return
	}

	// Create a new OpenAI client
	oClient := openai.NewClient(config.OpenAIToken)

	var oResponse OpenAIResponse
	schema, err := jsonschema.GenerateSchemaForType(oResponse)
	if err != nil {
		log.Fatalf("GenerateSchemaForType error: %v", err)
	}
	resp, err := oClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: openai.GPT4oMini,
		Messages: []openai.ChatCompletionMessage{
			{
				Role: openai.ChatMessageRoleSystem,
				Content: `Grade the provided assignment. 
				Ensure that students answered all the questions as listed below, and that the answers are approximately correct.
				Allow for minor variations in wording, differences in phrasing, and poor English.
				Return the answer in JSON format {Qs: [{q: 'a', g: 'correct'}, ...]}
				g must be one of ['correct', 'incorrect', 'more_details_needed', 'missing']

				Questions:
	> a. What is an _issue_? (generally anything that is true)

    > b. What is a _pull request_? (generally anything that is true)

    > c. How do I open up a _pull request_? (at least 3 steps)
		Fork the repository (if you don't have write access).
		Clone the repository to your local machine.
		Create a new branch for your changes.
		Make and commit your changes locally.
		Push the branch to your GitHub repository.
		Navigate to the original repository on GitHub.
		Click on "Pull Requests" and then "New Pull Request".
		Select the base and compare branches to review differences.
		Add a title and description, then submit the pull request.

    > d. Give me a step by step guide on how to add someone to your repository. (at least both going to repository settings and inviting a collaborator)
		Go to the repository on GitHub.
		Click on "Settings" in the repository menu.
		Select "Manage Access" or "Collaborators".
		Click "Invite a collaborator".
		Enter the collaborator's GitHub username or email address.

    > e. What is the difference between git and GitHub? (generally anything that is true)

    > f. What does git diff do? (generally anything that is true)

    > g. What is the main branch? (generally anything that is true)

    > h. Besides our initial commit if it is a new repository, should we directly push our changes directly into the main branch? (answer of no, with any 1 of the following)
		Better practice to create separate branches for new features or fixes.
		Using pull requests to merge into main allows for code review.
		Helps prevent untested or broken code from entering the main codebase.
		Facilitates collaboration and maintains code quality.
		In small or personal projects, pushing directly might be acceptable, but not ideal for team environments.`,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: content,
			},
		},
		ResponseFormat: &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "github_assignment_review",
				Schema: schema,
				Strict: true,
			},
		},
	})
	if err != nil {
		log.Print(err)
		http.Error(w, "error creating chat completion with OpenAI API", http.StatusInternalServerError)
		return
	}
	err = schema.Unmarshal(resp.Choices[0].Message.Content, &oResponse)
	if err != nil {
		log.Print(err)
		http.Error(w, "error unmarshalling OpenAI response", http.StatusInternalServerError)
		return
	}

	// format the answers as a markdown table
	var answers strings.Builder
	answers.WriteString("| Question | Grade |\n")
	answers.WriteString("| --- | --- |\n")
	for _, answer := range oResponse.Qs {
		answers.WriteString("| " + answer.Q + " | " + answer.G + " |\n")
	}

	// if all answers are correct, then approve the PR
	var correctAnswers int
	for _, thisAnswer := range oResponse.Qs {
		if thisAnswer.G == "correct" {
			correctAnswers++
		}
	}
	var response Response
	if correctAnswers == len(oResponse.Qs) {
		response = Response{
			Status:  APPROVE,
			Message: answers.String(),
		}
	} else {
		response = Response{
			Status:  REQUEST_CHANGES,
			Message: answers.String(),
		}
	}
	log.Println(response)
	err = createReview(ctx, ghToken, userName, repoName, prNum, response)
	if err != nil {
		log.Print(err)
		http.Error(w, "error creating review with GitHub API", http.StatusInternalServerError)
		return
	}

	writeHTTPResponse(w, response)
}

func writeHTTPResponse(w http.ResponseWriter, response Response) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func createReview(ctx context.Context, token string, userName string, repoName string, issue int, response Response) error {
	// Create the review body
	body := "### DSI Autograder\n" + response.Message

	// Create the review request object
	review := PullRequestReviewRequest{
		Body:  body,
		Event: response.Status,
	}

	log.Println(review)

	// Marshal to JSON
	reviewJSON, err := json.Marshal(review)
	if err != nil {
		return err
	}

	// Construct the URL
	url := fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%d/reviews", userName, repoName, issue)

	// Create the HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reviewJSON))
	if err != nil {
		return err
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "token "+token)

	// Create HTTP client
	client := &http.Client{}

	// Make the request
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check the response
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		// Success
		return nil
	} else {
		bodyBytes, _ := io.ReadAll(resp.Body)

		// Error
		return fmt.Errorf("GitHub API error: %s", string(bodyBytes))
	}
}

// ParseGitHubRepoURL validates the given URL and extracts the GitHub username and repository name.
func ParseGitHubRepoURL(urlStr string) (username string, reponame string, err error) {
	// Parse the URL to check if it's valid
	u, err := url.Parse(urlStr)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return "", "", errors.New("invalid URL")
	}

	// Ensure the URL has HTTP or HTTPS scheme
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", "", errors.New("URL must start with http or https")
	}

	// Check that the domain is github.com
	if strings.ToLower(u.Hostname()) != "github.com" {
		return "", "", errors.New("URL must have github.com as the domain")
	}

	// Split the path into components
	pathParts := strings.Split(strings.Trim(u.Path, "/"), "/")

	// Check if the path has at least two components: username and repository name
	if len(pathParts) < 2 {
		return "", "", errors.New("URL must include both username and repository name")
	}

	// Extract the username and repository name
	username = pathParts[0]
	reponame = pathParts[1]

	// Remove .git suffix if present
	reponame = strings.TrimSuffix(reponame, ".git")

	return username, reponame, nil
}

func staticHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, r.URL.Path[1:])
}
