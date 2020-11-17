#include "MySeamFinder.h"


pc::MySeamFinder::MySeamFinder(std::vector<cv::UMat> &maskMaps){
    for(int i=0; i<pc::numCamera; i++){
        cv::Mat temp;
        maskMaps[i].getMat(cv::ACCESS_FAST).copyTo(temp);
        maskMaps_.push_back(temp);
    }

    for(int i=0; i<pc::numCamera; i++){
        cv::Mat temp;
        cv::bitwise_and(maskMaps[i], maskMaps[(i+1)%pc::numCamera], temp, maskMaps[i]);
        overlapMaskMaps_.push_back(temp);

        std::vector<std::vector<int> > temp_vv;
        std::vector<int> temp_seampos;
        std::vector<int> temp_seamIntensity;
        for(int r=0; r<temp.rows; r++){
            temp_seampos.push_back(0);
            temp_seamIntensity.push_back(0);
            std::vector<int> temp_v;
            for(int c=0; c<temp.cols; c++){
                if(temp.at<uchar>(r, c-1) > 0 && temp.at<uchar>(r, c) > 0 ){
                    temp_v.push_back(c);
                }
            }
            temp_vv.push_back(temp_v);
        }
        overlapRowCols_.push_back(temp_vv);
        lastframe_seamPos_.push_back(temp_seampos);
        lastframe_seamItensity_.push_back(temp_seamIntensity);
        lastframe_seamPosMaps.push_back(cv::Mat::zeros(cv::Size(temp.cols, temp.rows), CV_8U));
    }

    for(int i=0; i<4; i++){
        cv::Mat temp(pc::resizeStitchResultSize, CV_8UC1);
        human_saliency.push_back(temp);
    }

    initialize_socket();
}

void pc::MySeamFinder::initialize_socket() {
    cli_sockfd=socket(AF_INET,SOCK_STREAM,0);/*创建连接的SOCKET */
    if(cli_sockfd<0){/*创建失败 */
        fprintf(stderr,"socker Error:%s\n",strerror(errno));
        exit(1);
    }
    int addrlen=sizeof(struct sockaddr_in);
    char seraddr[14]="127.0.0.1";
    struct sockaddr_in ser_addr,/* 服务器的地址*/ cli_addr;/* 客户端的地址*/
    bzero(&ser_addr,addrlen);
    ser_addr.sin_family=AF_INET;
    ser_addr.sin_addr.s_addr=inet_addr(seraddr);
    ser_addr.sin_port=htons(2222);
    if(connect(cli_sockfd,(struct sockaddr*)&ser_addr, addrlen)!=0)/*请求连接*/
    {
        /*连接失败 */
        fprintf(stderr,"Connect Error:%s\n",strerror(errno));
        close(cli_sockfd);
        exit(1);
    }
}

void pc::MySeamFinder::human_segmentation_socket(std::vector<cv::UMat> &remapImgs) {
    for(int i=0; i<4; i++){
        cv::Mat temp_mat = remapImgs[i].getMat(cv::ACCESS_FAST);
        unsigned  char* uch = temp_mat.data;
        send(cli_sockfd, uch, total_size,0);
        recv(cli_sockfd, recvb, 1024,0); /* 接受数据*/
        unsigned char* ptr = recvb+1024;
        int recived_bytes = 1024;
        while(recived_bytes<total_size/3){
            recv(cli_sockfd, ptr, 1024,0); /* 接受数据*/
            ptr +=1024;
            recived_bytes += 1024;
        }
        ptr = recvb;
        std::memcpy((void*) human_saliency[i].data, recvb, total_size/3*sizeof(unsigned char));
    }
}

void pc::MySeamFinder::translateTransform_x(cv::Mat const& src, cv::Mat& dst, int dx)   //向右平移x
{
    int rows = src.rows;
    int cols = src.cols;
    dst.create(rows, cols, src.type());
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            //平移后坐标映射到原图像
            int x = j - dx;
            //保证映射后的坐标在原图像范围内
            if (x >= 0&& x < cols)
                dst.at<uchar>(i,j) = src.at<uchar>(i,x);
        }
    }
}


void pc::MySeamFinder::translateTransform_y(cv::Mat const& src, cv::Mat& dst, int dy)   //向下平移y
{
    int rows = src.rows;
    int cols = src.cols;
    dst.create(rows, cols, src.type());
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            //平移后坐标映射到原图像
            int y = i - dy;
            //保证映射后的坐标在原图像范围内
            if (y >= 0&& y < rows)
                dst.at<uchar>(i,j) = src.at<uchar>(y,j);
        }
    }
}


bool pc::MySeamFinder::find_dp_temporal_fast(std::vector<cv::UMat> &remapImgs, std::vector<cv::UMat> &maskMapsSeam, bool human){
    clock_t time_end1=std::clock();

    int rows = remapImgs[0].rows;
    int cols = remapImgs[0].cols;
    bool flags4[4] = {1,1,1,1};
    if(not_firstframe){
        // 计算缝合线处差异，从而确定要不要更新缝合线
        bool flag = false; //不需要更新
        for(int i=0; i<pc::numCamera; i++){
            cv::Mat img1, img2;
            cv::cvtColor(remapImgs[i].getMat(cv::ACCESS_FAST), img1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(remapImgs[(i+1)%pc::numCamera].getMat(cv::ACCESS_FAST), img2, cv::COLOR_BGR2GRAY);

            int cnt_diff = 0, cur_diff, col;
            for(int r=0; r<rows; r++){
                col = lastframe_seamPos_[i][r];
                cur_diff = (img1.at<uchar>(r, col) + img2.at<uchar>(r, col))/2;
                if(std::abs(cur_diff - lastframe_seamItensity_[i][r])>0.05*cur_diff) cnt_diff++;
            }
            if(cnt_diff > 0.01*rows){
                flag = true;
                flags4[i]=1;
            } else
                flags4[i]=0;
        }
        if(!flag)
            return false;
    }

    std::vector<std::vector<int> > seamPos;
    std::vector<cv::Mat> remapImgs_gray_Mat;

    for(int i=0; i<4; i++){
        cv::Mat temp;
        cv::cvtColor(remapImgs[i].getMat(cv::ACCESS_FAST), temp, cv::COLOR_BGR2GRAY);
        remapImgs_gray_Mat.push_back(temp);

        if(!not_firstframe)
            seamPos.push_back(std::vector<int>(rows));
    }
    if(not_firstframe)
        seamPos = lastframe_seamPos_;

    uchar* ptmp = NULL;
    for(int i=0; i<4; i++){
        for(int r=0; r<pc::resizeStitchResultSize.height; r++){
            ptmp = remapImgs_gray_Mat[i].ptr<uchar>(r);
            for(int c=0; c<pc::resizeStitchResultSize.width; c++){
                spacial_cost[i][r][c] = INT32_MAX;
                spacial_mincost_lastcol[i][r][c] = 0;
                remapImgs_gray[i][r][c] = ptmp[c];
            }
        }
    }

    // compute saliency through gradient
    if(human== false)
        for(int i=0; i<4; i++){
            for(int r=1; r<pc::resizeStitchResultSize.height-1; r++){
                for(int c=1; c<pc::resizeStitchResultSize.width-1; c++){
                    saliency[i][r][c] = 10 * (std::abs(remapImgs_gray[i][r-1][c] - remapImgs_gray[i][r+1][c]) +
                                        std::abs(remapImgs_gray[i][r][c-1] - remapImgs_gray[i][r][c+1]));
                }
            }
        }
    else{

        human_segmentation_socket(remapImgs);

        uchar* ptmp = NULL;
        for(int i=0; i<4; i++){
            for(int r=0; r<pc::resizeStitchResultSize.height; r++){
                ptmp = human_saliency[i].ptr<uchar>(r);
                for(int c=0; c<pc::resizeStitchResultSize.width; c++){
                    saliency[i][r][c] = 1000*ptmp[c];
                }
            }
        }
    }

    for(int i=0; i<4; i++){
        int a = i, b = (i+1)%4;
        for(int r=0; r<pc::resizeStitchResultSize.height; r++){
            for(int c=0; c<pc::resizeStitchResultSize.width-1; c++){
                remapImgs_gray_diff_lr[i][r][c] = std::abs(remapImgs_gray[a][r][c] - remapImgs_gray[b][r][c+1]) +
                                                  std::abs(remapImgs_gray[a][r][c+1] - remapImgs_gray[b][r][c]);
            }
        }
    }
    for(int i=0; i<4; i++){
        int a = i, b = (i+1)%4;
        for(int r=0; r<pc::resizeStitchResultSize.height-1; r++){
            for(int c=0; c<pc::resizeStitchResultSize.width; c++){
                remapImgs_gray_diff_ud[i][r][c] = std::abs(remapImgs_gray[a][r+1][c] - remapImgs_gray[b][r][c]) +
                                                  std::abs(remapImgs_gray[a][r][c] - remapImgs_gray[b][r+1][c]);
            }
        }
    }

    //动态规划正向过程，计算所有可能路径的代价，并记录局部最优路径
    for(int i=0; i<pc::numCamera; i++){
        if(!flags4[i])
            continue;
        cv::Mat overlapMask_temp = overlapMaskMaps_[i];
        std::vector<std::vector<int> > overlap_eachrow_cols = overlapRowCols_[i];

        // first line
        int len = overlap_eachrow_cols[0].size();
        int col;
        for(int p=0; p<len; p++){
            col = overlap_eachrow_cols[0][p];
            spacial_cost[i][0][col] = remapImgs_gray_diff_lr[i][0][col];
            spacial_cost[i][0][col] += std::max(saliency[i][0][col], saliency[(i+1)%4][0][col]);
            if(not_firstframe)spacial_cost[i][0][col] += std::abs(col-lastframe_seamPos_[i][0])*0.1;  // temperal
        }

        // following lines
        for(int y=1; y<rows; y++){
            len = overlap_eachrow_cols[y].size();
            for(int p=0; p<len; p++){
                col = overlap_eachrow_cols[y][p]; //当前行的列
                int min_cost=INT32_MAX, cur_cost;
                int min_cost_col;
                // 遍历上一行所有点，计算当前点到上一行某点的代价，并记录最小值和对应位置
                int len1 = overlap_eachrow_cols[y-1].size();
                int col1, left_pos, right_pos;
                for(int p1=0; p1<len1; p1++){
                    col1 = overlap_eachrow_cols[y-1][p1];
                    if(col1-col<6 and col1-col>-6){
                        cur_cost = spacial_cost[i][y-1][col1];
                        cur_cost += remapImgs_gray_diff_lr[i][y][col1];
                        if(not_firstframe)cur_cost += std::abs(col-lastframe_seamPos_[i][y])*0.1;  // temperal
                        left_pos = std::min(col, col1);
                        right_pos = std::max(col, col1);
                        for(;left_pos<=right_pos; left_pos++){
                            cur_cost += std::max(saliency[i][y][left_pos], saliency[(i+1)%4][y][left_pos]);
                            cur_cost += remapImgs_gray_diff_ud[i][y][left_pos];
                            if(cur_cost>min_cost)
                                break;
                        }
                        if(cur_cost<min_cost){
                            min_cost = cur_cost;
                            min_cost_col = col1;
                        }
                    }
                }
                //记录最小值和对应位置
                spacial_cost[i][y][col] = min_cost;
                spacial_mincost_lastcol[i][y][col] = min_cost_col;
            }
        }

    }

    //动态规划逆向过程，在最后一行找出代价最小的，然后倒着根据局部最优路径找回去找到整条最优路径
    for(int i=0; i<pc::numCamera; i++){
        if(!flags4[i])
            continue;
        cv::Mat overlapMask_temp = overlapMaskMaps_[i];
        int min_cost_lastrow=INT32_MAX;
        int min_cost_lastrow_col=0;
        int len = overlapRowCols_[i][0].size();
        int col;
        for(int p=0; p<len; p++){
            col = overlapRowCols_[i][0][p];
            if(min_cost_lastrow > spacial_cost[i][rows-1][col]){
                min_cost_lastrow = spacial_cost[i][rows-1][col];
                min_cost_lastrow_col = col;
            }
        }

        seamPos[i][rows-1] = min_cost_lastrow_col;
        for(int r=rows-2; r>=0; r--){
            min_cost_lastrow_col = spacial_mincost_lastcol[i][r+1][min_cost_lastrow_col];
            seamPos[i][r] = min_cost_lastrow_col;
        }
    }

    std::vector<cv::Mat> maskMaps_temp(4);
    std::vector<cv::Mat> maskMapsSeam_temp(4);
    for(int i=0; i<pc::numCamera; i++) {
        maskMaps_temp[i] = maskMaps_[i];
        maskMapsSeam_temp[i] = maskMapsSeam[i].getMat(cv::ACCESS_WRITE);
        maskMapsSeam_temp[i].setTo(0);
    }

    for(int row=0; row<rows; row++){
        int col = 0;
        while(col<seamPos[0][row] && maskMaps_temp[1].at<uchar>(row, col) > 0){
            maskMapsSeam_temp[1].at<uchar>(row, col)=255;
            col++;
        }
        while(col<seamPos[3][row] && maskMaps_temp[0].at<uchar>(row, col) > 0){
            maskMapsSeam_temp[0].at<uchar>(row, col)=255;
            col++;
        }
        while(col<seamPos[2][row] && maskMaps_temp[3].at<uchar>(row, col) > 0){
            maskMapsSeam_temp[3].at<uchar>(row, col)=255;
            col++;
        }
        while(col<seamPos[1][row] && maskMaps_temp[2].at<uchar>(row, col) > 0){
            maskMapsSeam_temp[2].at<uchar>(row, col)=255;
            col++;
        }
        while(col<cols && maskMaps_temp[1].at<uchar>(row, col) > 0){
            maskMapsSeam_temp[1].at<uchar>(row, col)=255;
            col++;
        }
    }
// save last frame seam
    for(int i=0; i<pc::numCamera; i++){
        if(!flags4[i])
            continue;
        for(int row=0; row<rows; row++){
            int col = seamPos[i][row];
            lastframe_seamPos_[i][row] = col;
            lastframe_seamItensity_[i][row] = (remapImgs_gray[i][row][col] + remapImgs_gray[(i+1)%4][row][col])/2;
        }
    }
    not_firstframe=true;

    return true;
}