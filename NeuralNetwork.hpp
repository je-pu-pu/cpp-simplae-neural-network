#include <vector>
#include <limits>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include <iostream>

/**
 * ニューラルネットワーク層
 *
 */
class NeuralNetworkLayer
{
public:
	typedef float ValueType;
	typedef float WeightType;
	typedef std::vector< ValueType > ValueList;
	typedef std::vector< WeightType > WeightList;
	
	static const ValueType BiasValue = 1;
	
private:
	ValueList value_list_;				/// ニューロンの値の一覧
	WeightList weight_list_;			/// 次の層へのウエイトの一覧
	WeightList bias_weight_list_;		/// 次の層へのバイアス値のウエイトの一覧
	ValueList error_list_;				/// エラー値の一覧
	
	ValueList instruction_signal_list_;	/// 教師信号の値の一覧
	
	NeuralNetworkLayer* next_layer_;
	
	/**
	 * シグモイド関数
	 *
	 * @param value 入力値
	 * @return 出力値
	 */
	ValueType output_function( ValueType value )
	{
		return 1 / ( 1 + std::exp( -value ) );
	}
	
	/**
	 * 指定した範囲内のランダムな値を返す
	 *
	 * @param min 最小値
	 * @param max 最大値
	 * @return 指定した範囲内のランダムな値を返す
	 */
	WeightType get_random_weight( WeightType min = -1, WeightType  max = 1 )
	{
		return min + ( static_cast< WeightType >( rand() ) / RAND_MAX ) * ( max - min );
	}
	
public:
	NeuralNetworkLayer( size_t size )
		: value_list_( size, 0 )
		, error_list_( size, 0 )
		, instruction_signal_list_( size, 0 )
		, next_layer_( 0 )
	{
		
	}
	
	size_t size() { return value_list_.size(); }
	
	void set_next_layer( NeuralNetworkLayer* next_layer )
	{
		next_layer_ = next_layer;
		weight_list_.resize( size() * next_layer_->size() );
		bias_weight_list_.resize( next_layer_->size() );
	}
	
	void randomize_weight()
	{
		// std::mt19377 r;
		// std::uniform_real_distribution<> d( -1.f, 1.f );
		
		for ( auto i = weight_list_.begin(); i != weight_list_.end(); ++i )
		{
			*i = get_random_weight();
		}
		
		for ( auto i = bias_weight_list_.begin(); i != bias_weight_list_.end(); ++i )
		{
			*i = get_random_weight();
		}
	}
	
	void set_value( size_t n, ValueType value )
	{
		value_list_[ n ] = value;
	}
	
	ValueType get_value( size_t n ) const
	{
		return value_list_[ n ];
	}
	
	void set_instruction_signal( size_t n, ValueType value )
	{
		instruction_signal_list_[ n ] = value;
	}
	
	size_t get_max_value_index() const
	{
		size_t max_index = 0;
		ValueType max = -std::numeric_limits< ValueType >::max();
		
		// std::cout << "---" << std::endl;
		
		for ( size_t n = 0; n < value_list_.size(); n++ )
		{
			// std::cout << n << " : " << value_list_[ n ] << std::endl;
			
			if ( value_list_[ n ] > max )
			{
				max = value_list_[ n ];
				max_index = n;
			}
		}
		
		// std::cout << "res : " << max_index << std::endl;
		
		// std::cout << "---" << std::endl;
		
		return max_index;
	}
	
	void forward_propagation()
	{
		for ( size_t n = 0; n < next_layer_->value_list_.size(); n++ )
		{
			ValueType value = 0;
			
			for ( size_t m = 0; m < value_list_.size(); m++ )
			{
				value += value_list_[ m ] * weight_list_[ m * next_layer_->size() + n ];
			}
			
			value += BiasValue * bias_weight_list_[ n ];
			
			next_layer_->value_list_[ n ] = output_function( value );
		}
	}
	
	void calculate_error_list()
	{
		if ( next_layer_ )
		{
			for ( size_t n = 0; n < value_list_.size(); n++ )
			{
				ValueType e = 0;
				
				for ( size_t m = 0; m < next_layer_->value_list_.size(); m++ )
				{
					e += next_layer_->error_list_[ m ] * weight_list_[ n * next_layer_->size() + m ];
				}
				
				error_list_[ n ] = e * value_list_[ n ] * ( 1 - value_list_[ n ] );
			}
		}
		else
		{
			for ( size_t n = 0; n < value_list_.size(); n++ )
			{
				error_list_[ n ] = ( instruction_signal_list_[ n ] - value_list_[ n ] ) * value_list_[ n ] * ( 1 - value_list_[ n ] );
			}
		}
	}
	
	void back_propagation( ValueType learning_rate )
	{
		for ( size_t n = 0; n < value_list_.size(); n++ )
		{
			for ( size_t m = 0; m < next_layer_->value_list_.size(); m++ )
			{
				weight_list_[ n * next_layer_->size() + m ] += learning_rate * next_layer_->error_list_[ m ] * value_list_[ n ];
			}
		}
		
		for ( size_t n = 0; n < bias_weight_list_.size(); n++ )
		{
			bias_weight_list_[ n ] += learning_rate * next_layer_->error_list_[ n ] * BiasValue;
		}
	}
	
	/**
	 * 平均二乗誤差を計算して返す
	 *
	 * @return float 平均二乗誤差
	 */
	ValueType calculate_error() const
	{
		ValueType error = 0;
		
		for ( size_t n = 0; n < value_list_.size(); n++ )
		{
			error += pow( value_list_[ n ] - instruction_signal_list_[ n ], 2 );
		}
		
		return error / value_list_.size();
	}
	
	friend std::istream& operator >> ( std::istream& in, NeuralNetworkLayer& layer )
	{
		in.read( reinterpret_cast< char* >( & layer.weight_list_[ 0 ] ), sizeof( WeightType ) * layer.weight_list_.size() );
		in.read( reinterpret_cast< char* >( & layer.bias_weight_list_[ 0 ] ), sizeof( WeightType ) * layer.bias_weight_list_.size() );
		
		return in;
	}
	
	friend std::ostream& operator << ( std::ostream& out, NeuralNetworkLayer& layer )
	{
		out.write( reinterpret_cast< const char* >( & layer.weight_list_[ 0 ] ), sizeof( WeightType ) * layer.weight_list_.size() );
		out.write( reinterpret_cast< const char* >( & layer.bias_weight_list_[ 0 ] ), sizeof( WeightType ) * layer.bias_weight_list_.size() );
		
		return out;
	}
};

/**
 * ニューラルネットワーク
 *
 */
class NeuralNetwork
{
public:
	typedef float ValueType;
	
private:
	NeuralNetworkLayer input_layer_;
	NeuralNetworkLayer hidden_layer_;
	NeuralNetworkLayer output_layer_;
	
	ValueType learning_rate_;
	
public:
	/**
	 * コンストラクタ
	 *
	 * @param ni 入力層のニューロン数
	 * @param nh 隠れ層のニューロン数
	 * @param no 出力層のニューロン数
	 */
	NeuralNetwork( size_t ni, size_t nh, size_t no )
		: input_layer_( ni )
		, hidden_layer_( nh )
		, output_layer_( no )
		, learning_rate_( 0.01 )
	{
		srand( time( 0 ) );
		
		input_layer_.set_next_layer( & hidden_layer_ );
		hidden_layer_.set_next_layer( & output_layer_ );
		
		randomize_weight();
	}
	
	void randomize_weight()
	{
		input_layer_.randomize_weight();
		hidden_layer_.randomize_weight();
	}
	
	/**
	 * 学習率を設定する
	 *
	 */
	void set_learning_rate( ValueType rate )
	{
		learning_rate_ = rate;
	}
	
	/**
	 * 入力層のニューロンの値を設定する
	 *
	 * @param n 値を設定するニューロンのインデックス
	 * @param value ニューロンの値
	 */
	void set_input( size_t n, ValueType value )
	{
		input_layer_.set_value( n, value );
	}
	
	/**
	 * 出力層のニューロンの値を取得する
	 *
	 * @param n 値を取得するニューロンのインデックス
	 * @return ニューロンの値
	 */
	ValueType get_output( size_t n ) const
	{
		return output_layer_.get_value( n );
	}
	
	/**
	 * 教師信号の値を設定する
	 *
	 * @param n ニューロンのインデックス
	 * @param value ニューロンの値
	 */
	void set_instruction_signal( size_t n, ValueType value )
	{
		output_layer_.set_instruction_signal( n, value );
	}
	
	/**
	 * 出力層のニューロンのうち値が最大のニューロンのインデックスを取得する
	 *
	 * @return 値が最大のニューロンのインデックス
	 */
	size_t get_max_output_index() const
	{
		return output_layer_.get_max_value_index();
	}
	
	/**
	 * 平均二乗誤差を計算して返す
	 *
	 * @return float 平均二乗誤差
	 */
	ValueType calculate_error() const
	{
		return output_layer_.calculate_error();
	}
	
	/**
	 * 
	 *
	 */
	void forward_propagation()
	{
		input_layer_.forward_propagation();
		hidden_layer_.forward_propagation();
	}
	
	/**
	 * 誤差逆伝播
	 *
	 */
	void back_propagation()
	{
		output_layer_.calculate_error_list();
		hidden_layer_.calculate_error_list();
		
		hidden_layer_.back_propagation( learning_rate_ );
		input_layer_.back_propagation( learning_rate_ );
	}
	
	bool load( const char* file_path )
	{
		std::ifstream in( file_path, std::ios::binary );
		
		if ( ! in.is_open() )
		{
			return false;
		}
		
		in >> input_layer_ >> hidden_layer_ >> output_layer_;
		
		return true;
	}
	
	bool save( const char* file_path )
	{
		std::ofstream out( file_path, std::ios::binary );
		
		if ( ! out.is_open() )
		{
			return false;
		}
		
		out << input_layer_ << hidden_layer_ << output_layer_;
		
		return true;
	}
};

