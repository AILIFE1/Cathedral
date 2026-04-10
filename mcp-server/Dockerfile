FROM python:3.12-slim

RUN pip install --no-cache-dir cathedral-mcp==1.0.1

# Glama verification key - server starts and responds to MCP introspection
# Replace with your real key from cathedral-ai.com
ENV CATHEDRAL_API_KEY=glama-check

CMD ["cathedral-mcp"]
