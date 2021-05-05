#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>
 
 
#include <filesystem>
#include <iostream>


int main()
{
    //std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::array<int, 10> v{1,2,3,4,5,6,7,8,9,10};
 
    int sum = std::accumulate(&v[0], &v[10], 0);
 
    int product = std::accumulate(&v[0], &v[3], 1, std::multiplies<int>());
 
    auto dash_fold = [](std::string a, int b) {
                         return std::move(a) + '-' + std::to_string(b);
                     };
 
    std::string s = std::accumulate(std::next(v.begin()), v.end(),
                                    std::to_string(v[0]), // start with first element
                                    dash_fold);
 
    // Right fold using reverse iterators
    std::string rs = std::accumulate(std::next(v.rbegin()), v.rend(),
                                     std::to_string(v.back()), // start with last element
                                     dash_fold);
 
    std::cout << "sum: " << sum << '\n'
              << "product: " << product << '\n'
              << "dash-separated string: " << s << '\n'
              << "dash-separated string (right-folded): " << rs << '\n';
              
              
    std::string f="Alexnet_Save//cnn_data/alexnet_model/conv1_w.npy";
    auto len=f.length();
    auto index=f.find_last_of('/');
    bool path=true;
    if (index == std::string::npos) {
        path=false;
    }
    std::cout<<"Total:"<<f<<std::endl;
    if(path)
    	std::cout<<"Path:"<<f.substr(0, index)<<std::endl;
    std::cout<<"filename:"<<f.substr(index + 1, len - index)<<std::endl;
    
    std::system(("mkdir -p "+f.substr(0,index)).c_str());
    
   
    
    
}
