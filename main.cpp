#include "NeuralNetwork.hpp"
#include <iostream>

float r( float min = -1, float max = 1 )
{
	return min + ( static_cast< float >( rand() ) / RAND_MAX ) * ( max - min );
}

float f( float x )
{
	return x * x;
}

int main( int, char** )
{
	NeuralNetwork nn( 1, 100, 1 );
	nn.load( "nn.dat" );
	
	float e_sum = 0.f;
	
	for ( int n = 0; n < 10000000; n++ )
	{
		float x = r( 0, 1 );
		float y = f( x );
		
		nn.set_input( 0, x );
		nn.set_instruction_signal( 0, y );
		nn.forward_propagation();
		nn.back_propagation();
		
		float z = nn.get_output( 0 );
		
		// std::cout << x << " : " << nn.get_output( 0 ) << " : " << f( x ) << " : " << nn.calculate_error() << std::endl;
		
		float e = nn.calculate_error();
		e_sum += e;
		
		printf( "%08d : %8.12f -> %8.12f : %8.12f : %8.12f : %8.12f\n", n, x, z, y, e, e_sum / ( n + 1 ) );
	}
	
	nn.save( "nn.dat" );
	
	return 0;
}