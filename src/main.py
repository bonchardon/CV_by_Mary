from asyncio import run
import geopandas as gpd

from core.comp_vision import ImageIdentification


async def main() -> None:
    return await ImageIdentification().dataset_prep()


if __name__ == '__main__':
    run(main())
