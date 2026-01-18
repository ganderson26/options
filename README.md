# options
Various scripts for Options

# GoDaddy
ssh jejtxlk4zmlg@50.63.7.156 -p 22
Marathon#262

python3 --version
3.12.3

cd options
ls -a

source virtualenv/options/3.9/bin/activate

python3 yahoo_option_chains.py

429 Client Error: Too Many Requests for url: https://query2.finance.yahoo.com

# GitHub
(base) ganderson@Glenns-MacBook-Pro-3.local:~/codebase/options-examples/options$git add .
(base) ganderson@Glenns-MacBook-Pro-3.local:~/codebase/options-examples/options$git commit -m "Latest code 12/23/2025"
(base) ganderson@Glenns-MacBook-Pro-3.local:~/codebase/options-examples/options$git push -u -f origin main

Enumerating objects: 155, done.
Counting objects: 100% (155/155), done.
Delta compression using up to 16 threads
Compressing objects: 100% (143/143), done.
Writing objects: 100% (144/144), 5.63 MiB | 3.61 MiB/s, done.
Total 144 (delta 40), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (40/40), completed with 8 local objects.
remote: error: GH013: Repository rule violations found for refs/heads/main.
remote: 
remote: - GITHUB PUSH PROTECTION
remote:   —————————————————————————————————————————
remote:     Resolve the following violations before pushing again
remote: 
remote:     - Push cannot contain secrets
remote: 
remote:     
remote:      (?) Learn how to resolve a blocked push
remote:      https://docs.github.com/code-security/secret-scanning/working-with-secret-scanning-and-push-protection/working-with-push-protection-from-the-command-line#resolving-a-blocked-push
remote:     
remote:      (?) This repository does not have Secret Scanning enabled, but is eligible. Enable Secret Scanning to view and manage detected secrets.
remote:      Visit the repository settings page, https://github.com/ganderson26/options/settings/security_analysis
remote:     
remote:     
remote:       —— OpenAI API Key ————————————————————————————————————
remote:        locations:
remote:          - commit: 8b0ff351d33f28dc3f4ada5dec141ed95a6bf090
remote:            path: src/chat-sentiment.py:3
remote:          - commit: 8b0ff351d33f28dc3f4ada5dec141ed95a6bf090
remote:            path: src/chat-sentiment.py:25
remote:     
remote:        (?) To push, remove secret from commit(s) or follow this URL to allow the secret.
remote:        https://github.com/ganderson26/options/security/secret-scanning/unblock-secret/37GKgRmp9Kyl8Ay9FsaRjtlYKQL
remote:     
remote: 
remote: 
To github.com:ganderson26/options.git
 ! [remote rejected] main -> main (push declined due to repository rule violations)
error: failed to push some refs to 'github.com:ganderson26/options.git'

Use that URL above and allow secrets

(base) ganderson@Glenns-MacBook-Pro-3.local:~/codebase/options-examples/options$git push -u -f origin main
Enumerating objects: 155, done.
Counting objects: 100% (155/155), done.
Delta compression using up to 16 threads
Compressing objects: 100% (143/143), done.
Writing objects: 100% (144/144), 5.63 MiB | 4.74 MiB/s, done.
Total 144 (delta 42), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (42/42), completed with 8 local objects.
To github.com:ganderson26/options.git
   85f8a65..8b0ff35  main -> main
branch 'main' set up to track 'origin/main'.

