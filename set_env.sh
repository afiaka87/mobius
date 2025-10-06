source .env

echo "$ANTHROPIC_API_KEY"
echo "$OPENAI_API_KEY"
echo "$FAL_KEY"

fly secrets set \
	ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
	OPENAI_API_KEY="$OPENAI_API_KEY" \
        FAL_KEY="$FAL_KEY" \
	--app mobius-api



