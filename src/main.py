from asyncio import run

from core.comp_vision import ImageIdentification


async def main() -> None:
    return await ImageIdentification().run()


if __name__ == '__main__':
    run(main())
