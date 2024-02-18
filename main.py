from arguments import args
from utils import get_app, get_pipelines

# RUNNING CHAT
#####################################################################################
app = get_app()

if __name__ == "__main__":
    pipelines = get_pipelines()

    if args.dev:
        while True:
            print("====================")
            inputs = input("Enter input text ('q' for exit)): ")
            if inputs == "q":
                break

            res = pipelines["query_pipeline"](query=inputs, debug=True)
            print("Answer:", res["answers"])

    else:
        import logging
        import uvicorn

        logging.basicConfig(
            format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
        )
        logger = logging.getLogger(__name__)
        logging.getLogger("elasticsearch").setLevel(logging.WARNING)
        logging.getLogger("haystack").setLevel(logging.INFO)

        logger.info("Open http://127.0.0.1:8000/docs to see Swagger API Documentation.")
        logger.info(
            "Or just try it out directly: curl --request POST --url 'http://127.0.0.1:8000/query' "
            '-H "Content-Type: application/json"  --data \'{"query": "Who is the father of Arya Stark?"}\''
        )

        uvicorn.run(app, host="0.0.0.0", port=8000)
