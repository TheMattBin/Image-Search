# Z.AI GLM Code Review Setup Guide

## 🚀 Using Z.AI Instead of Claude

Good choice! Z.AI's GLM models are much more cost-effective while still providing excellent code reviews.

---

## 📋 Setup Steps

### 1. Get Z.AI API Key

1. Visit https://open.bigmodel.cn/
2. Sign up / Log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with something like `sk-...`)

### 2. Add API Key to GitHub

1. Go to your repository on GitHub
2. **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `ZAI_API_KEY`
5. Value: Paste your Z.AI API key
6. Click **Add secret**

### 3. Test the Workflow

Create a test PR:

```bash
git checkout -b feature/test-zai-review
echo "# Test Z.AI review" >> README.md
git add README.md
git commit -m "Test Z.AI GLM code review"
git push origin feature/test-zai-review
```

---

## 🔧 Configuration

### Workflow File

The workflow is in `.github/workflows/zai-review.yml`

**Key Settings:**

```yaml
# Model used (you can change this)
"model": "glm-4-flash"

# Available GLM models:
# - glm-4-flash     (fastest, cheapest)
# - glm-4-air       (balanced)
# - glm-4-plus      (more capable)
# - glm-4           (most capable, slower)
```

**To change model:**

Edit `.github/workflows/zai-review.yml`:

```yaml
"d": '{
  "model": "glm-4-plus",  # Change this
  ...
}'
```

### API Endpoint

The workflow uses: `https://open.bigmodel.cn/api/paas/v4/chat/completions`

This is Z.AI's standard API endpoint. If this changes, update the workflow.

---

## 🎯 Review Focus Areas

The GLM model reviews code for:

1. **🔴 Security Vulnerabilities**
   - Injection attacks
   - Authentication issues
   - XSS vulnerabilities
   - File handling problems

2. **🟡 Code Quality**
   - Unused code
   - Complex logic
   - Poor error handling
   - Missing validation

3. **⚡ Performance Issues**
   - Inefficient algorithms
   - Unnecessary computations
   - Memory leaks

4. **🐛 Potential Bugs**
   - Edge cases
   - Null handling
   - Race conditions
   - Validation errors

---

## 💰 Cost Comparison

### Claude (Previous)
- **Claude Sonnet 4**: ~$3 per million input tokens
- Typical PR review: $0.05-$0.15

### Z.AI GLM (Current)
- **GLM-4-Flash**: ~$0.1 per million tokens
- **GLM-4-Plus**: ~$0.5 per million tokens
- Typical PR review: $0.001-$0.01

**Savings: ~10-100x cheaper! 🎉**

---

## 🐛 Troubleshooting

### Review Not Running?

**Check 1: API Key**
```bash
# Verify secret is set
gh secret list
```

**Check 2: Workflow Logs**
1. Go to Actions tab
2. Click on failed run
3. Expand "Review with Z.AI GLM" step
4. Look for error messages

**Check 3: API Format**
Z.AI might have different response format. Check the debug output:

```bash
# In workflow logs, look for:
# "API Response:" section
```

If the format is different, you may need to adjust this line in the workflow:

```yaml
REVIEW_TEXT=$(echo "$REVIEW" | jq -r '.choices[0].message.content // ...')
```

### Common Issues

**Issue 1: 401 Unauthorized**
- Problem: Invalid API key
- Solution: Regenerate key and update secret

**Issue 2: Rate Limiting**
- Problem: Too many requests
- Solution: Add `retry` logic or use `glm-4-flash` (higher limits)

**Issue 3: Empty Review**
- Problem: API response format changed
- Solution: Check logs for actual response format and update parsing

---

## 🔄 Switching Models

### For Faster Reviews (Cheaper)
```yaml
"model": "glm-4-flash"
```
- Speed: ⚡⚡⚡
- Quality: ⭐⭐⭐
- Cost: 💰

### For Balanced Performance
```yaml
"model": "glm-4-air"
```
- Speed: ⚡⚡
- Quality: ⭐⭐⭐⭐
- Cost: 💰💰

### For Best Quality
```yaml
"model": "glm-4-plus"
```
- Speed: ⚡
- Quality: ⭐⭐⭐⭐⭐
- Cost: 💰💰💰

---

## 📊 Monitoring

### Track Usage

Check your Z.AI dashboard:
1. Visit https://open.bigmodel.cn/
2. Go to Usage section
3. Monitor token consumption
4. Set up alerts if needed

### Cost Optimization

**Tips to reduce costs:**
1. Use `glm-4-flash` for initial reviews
2. Only use higher models for critical PRs
3. Review large PRs manually (>2000 lines)
4. Cache results when possible

---

## 🎉 Summary

**You're now using Z.AI's GLM for automated code reviews!**

✅ **Cheaper** than Claude (10-100x savings)
✅ **Fast** reviews with GLM-4-Flash
✅ **High quality** code analysis
✅ **Same workflow** - nothing else changes

**Configuration:**
- API Secret: `ZAI_API_KEY` (in GitHub Secrets)
- Model: `glm-4-flash` (adjustable)
- Label: `glm-reviewed`

**Ready to save money while catching bugs! 🚀**
