.PHONY: check-codestyle codestyle

check-codestyle:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	pre-commit run
