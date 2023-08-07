/*!
\file
\brief File contains Map class implementation.
*/
#include <iostream>
#include "simple_toposlam_model/map.hpp"

Map::Map()
{
    cellSize = 0;
    height = 0;
    width = 0;
    unknownIsObstacle = false;
}


Map::Map(const Map &&obj)
{
    cellSize = obj.cellSize;
    height = obj.height;
    width = obj.width;
    grid = std::move(obj.grid);
    unknownIsObstacle = obj.unknownIsObstacle;
}


bool Map::CellIsObstacle(int i, int j) const
{
    //std::cout << " obst " << (int) grid[(height - 1 - i) * width + j] << "\n";
    if (unknownIsObstacle)
    {
        return (grid[i * width + j] != 0);
    }
    return (grid[i * width + j] > 0);
}


bool Map::CellIsUnknown(int i, int j) const
{
    return (grid[i * width + j] < 0);
}


bool Map::CellIsTraversable(int i, int j) const
{
    return (grid[i * width + j] == 0);
}


bool Map::CellOnGrid(int i, int j) const
{
    return (i < height && i >= 0 && j < width && j >= 0);
}

unsigned int Map::GetHeight() const
{
    return height;
}


unsigned int Map::GetWidth() const
{
    return width;
}


float Map::GetCellSize() const
{
    return cellSize;
}


Node Map::GetClosestNode(const Point &point) const
{
    Node res;
    res.i = static_cast<int>(((point.Y() - originPosition.Y()) / cellSize));
    res.j = static_cast<int>(((point.X() - originPosition.X()) / cellSize));

    if(res.i < 0)
    {
        res.i = 0;
    }
    if(res.i > height - 1)
    {
        res.i = height - 1;
    }
    if(res.j < 0)
    {
        res.j = 0;
    }
    if(res.j > width - 1)
    {
        res.j = width - 1;
    }

    return res;
}


Point Map::GetPoint(const Node &node) const
{

    auto pos_x = originPosition.X() + (node.j + 0.5f) * cellSize;
    auto pos_y = originPosition.Y() + (node.i + 0.5f) * cellSize;

    return {pos_x, pos_y};
}

Map& Map::operator= (const Map &&obj)
{
    if(this != &obj)
    {
        cellSize = obj.cellSize;
        height = obj.height;
        width = obj.width;
        grid = std::move(obj.grid);
    }
    return *this;
}


 void Map::Update(float cellSize, size_t height, size_t width, Point originPos, Quaternion originOr, const std::vector<signed char> &grid)
 {
    this->cellSize = cellSize;
    this->height = height;
    this->width = width;
    this->originPosition = originPos;
    this->originOrientation = originOr;
    this->grid = grid;
 }