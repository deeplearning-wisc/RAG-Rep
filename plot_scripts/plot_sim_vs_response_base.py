from understanding_rag.plotting import main


if __name__ == "__main__":
    import sys

    sys.argv.insert(1, "sim-vs-response")
    sys.argv.append("--base")
    main()

