from intraday_loader import smoke_test_connections


def main() -> None:
    checks = smoke_test_connections()
    for ok, message in checks:
        print(("[PASS] " if ok else "[FAIL] ") + message)

    if not all(ok for ok, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
